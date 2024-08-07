"""Training script for detector."""

import argparse
import copy
import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pycocotools.cocoeval import COCOeval
from torch import nn
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
import json

import utils
import wandb
from detector_b import Detector

NUM_CATEGORIES = 15
VALIDATION_ITERATION = 250  # 250
NUM_ITERATIONS = 6000  # 10000
LEARNING_RATE = 1e-4 # 1e-4
WEIGHT_CAT = 1
WEIGHT_POS = 1
WEIGHT_NEG = 1
WEIGHT_REG = 1  # Regression loss
BATCH_SIZE = 8


def compute_loss(
    prediction_batch: torch.Tensor, target_batch: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Compute loss between predicted tensor and target tensor.

    Args:
        prediction_batch: Batched predictions. Shape (N,C,H,W).
        target_batch: Batched targets. shape (N,C,H,W).

    Returns:
        Tuple of three separate loss terms:
            reg_mse: Mean squared error of regression targets.
            pos_mse: Mean squared error of positive confidence channel.
            neg_mse: Mean squared error of negative confidence channel.
    """
    # positive / negative indices
    pos_indices = torch.nonzero(target_batch[:, 4, :, :] == 1, as_tuple=True)
    neg_indices = torch.nonzero(target_batch[:, 4, :, :] == 0, as_tuple=True)

    pred_cat = prediction_batch[:, 5:, :, :]
    target_cat = target_batch[:, 5:, :, :]

    # compute loss
    reg_mse = nn.functional.mse_loss(
        prediction_batch[pos_indices[0], 0:4, pos_indices[1], pos_indices[2]],
        target_batch[pos_indices[0], 0:4, pos_indices[1], pos_indices[2]],
    )
    pos_mse = nn.functional.mse_loss(
        prediction_batch[pos_indices[0], 4, pos_indices[1], pos_indices[2]],
        target_batch[pos_indices[0], 4, pos_indices[1], pos_indices[2]],
    )
    neg_mse = nn.functional.mse_loss(
        prediction_batch[neg_indices[0], 4, neg_indices[1], neg_indices[2]],
        target_batch[neg_indices[0], 4, neg_indices[1], neg_indices[2]],
    )
    cat_loss = nn.functional.cross_entropy(pred_cat,target_cat)
    return reg_mse, pos_mse, neg_mse, cat_loss


def train(device: str = "cpu") -> None:
    """Train the network.

    Args:
        device: The device to train on.
    """
    wandb.init(project="detector_baseline")

    # Init model
    detector = Detector().to(device)

    wandb.watch(detector)

    input_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    annotation_path = "./dd2419_23_data/dd2419_23_data_b/annotations/training.json"
    with open(annotation_path) as f:
        data = json.load(f)
    cat_dict = data['categories']

    # dataset = CocoDetection(
    #     root="./detectorNN/dd2419_23_data_a/training",
    #     annFile="./detectorNN/dd2419_23_data_a/annotations/merged_training640.json",
    #     transforms=input_transforms,
    # )
    # dataset = wrap_dataset_for_transforms_v2(dataset)
    # val_dataset = CocoDetection(
    #     root="./detectorNN/dd2419_23_data_a/validation",
    #     annFile="./detectorNN/dd2419_23_data_a/annotations/merged_validation.json",
    #     transforms=input_transforms,  # make sure not to accidentally augment validation data
    # )

    dataset = CocoDetection(
        root="./dd2419_23_data/dd2419_23_data_b/train",
        annFile="./dd2419_23_data/dd2419_23_data_b/annotations/train_b.json",
        transforms=input_transforms,
    )
    dataset = wrap_dataset_for_transforms_v2(dataset)
    val_dataset = CocoDetection(
        root="./dd2419_23_data/dd2419_23_data_b/validation",
        annFile="./dd2419_23_data/dd2419_23_data_b/annotations/test_b.json",
        transforms=input_transforms,  # make sure not to accidentally augment validation data
    )
    val_dataset = wrap_dataset_for_transforms_v2(val_dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    # training params
    wandb.config.max_iterations = NUM_ITERATIONS
    wandb.config.learning_rate = LEARNING_RATE
    wandb.config.weight_pos = WEIGHT_POS
    wandb.config.weight_neg = WEIGHT_NEG
    wandb.config.weight_reg = WEIGHT_REG

    # run name (to easily identify model later)
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_name = wandb.config.run_name = "det_{}".format(time_string)

    # init optimizer
    optimizer = torch.optim.Adam(
        detector.parameters(), lr=LEARNING_RATE)  # Tune the LEARNING_RATE

    # load test images
    # these will be evaluated in regular intervals
    test_images = []
    show_test_images = False
    directory = "./dd2419_23_data/dd2419_23_data_b/testImage"  # "./test_images"  #
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            test_images.append(Image.open(file_path))

    if test_images:
        show_test_images = True
        test_image_batch = torch.stack(input_transforms(test_images)).to(device)

    print("Training started...")

    current_iteration = 1
    while current_iteration <= NUM_ITERATIONS:
        for images, anns in dataloader:
            # convert bounding boxes to target tensor
            target_batch = detector.anns_to_target_batch(anns)

            img_batch = torch.stack(images).to(device)
            target_batch = target_batch.to(device)

            # run network
            out = detector(img_batch)

            # 4. COMPUTE LOSS
            reg_mse, pos_mse, neg_mse, cat_loss = compute_loss(out, target_batch)
            loss = WEIGHT_POS * pos_mse + WEIGHT_REG * reg_mse + WEIGHT_NEG * neg_mse + WEIGHT_CAT * cat_loss

            # optimize (Update network params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "total loss": loss.item(),
                    "loss pos": pos_mse.item(),
                    "loss neg": neg_mse.item(),
                    "loss reg": reg_mse.item(),
                    "loss cat": cat_loss.item(),
                },
                step=current_iteration,
            )

            print(
                "Iteration: {}, loss: {}".format(current_iteration, loss.item()),
            )

            # Validate every N iterations
            if current_iteration % VALIDATION_ITERATION == 1:
                validate(detector, val_dataloader, current_iteration, device)

            # generate visualization every N iterations
            if current_iteration % VALIDATION_ITERATION == 1 and show_test_images:
                detector.eval()
                with torch.no_grad():
                    out = detector(test_image_batch).cpu()
                    bbs = detector.out_to_bbs(out, 0.5)

                    for i, test_image in enumerate(test_images):
                        # add bounding boxes
                        result_image = utils.draw_detections(
                            test_image,
                            bbs[i],
                            confidence=out[i, 4, :, :],
                            category_dict=cat_dict,
                            channel_first=True,
                        )
                        wandb.log(
                            {"test_img_{i}".format(i=i): wandb.Image(result_image)},
                            step=current_iteration,
                        )
                        plt.close()
                detector.train()

            current_iteration += 1
            if current_iteration > NUM_ITERATIONS:
                break

    print("\nTraining completed (max iterations reached)")

    model_path = "{}.pt".format(run_name)
    utils.save_model(detector, model_path)
    wandb.save(model_path)

    print("Model weights saved at {}".format(model_path))


def validate(
    detector: Detector,
    val_dataloader: torch.utils.data.DataLoader,
    current_iteration: int,
    device: str,
) -> None:
    """Compute validation metrics and log to wandb.

    Args:
        detector: The detector module to validate.
        val_dataloader: The dataloader for the validation dataset.
        current_iteration: The current training iteration. Used for logging.
        device: The device to run validation on.
    """
    detector.eval()
    coco_pred = copy.deepcopy(val_dataloader.dataset.coco)
    coco_pred.dataset["annotations"] = []
    with torch.no_grad():
        count = total_pos_mse = total_reg_mse = total_neg_mse = loss = total_cat_loss= 0
        image_id = ann_id = 0
        for val_images, val_anns in val_dataloader:
            val_img_batch = torch.stack(val_images).to(device)
            val_target_batch = detector.anns_to_target_batch(val_anns).to(device)
            val_out = detector(val_img_batch)
            reg_mse, pos_mse, neg_mse, cat_loss = compute_loss(val_out, val_target_batch)
            total_reg_mse += reg_mse
            total_pos_mse += pos_mse
            total_neg_mse += neg_mse
            total_cat_loss += cat_loss
            loss += WEIGHT_POS * pos_mse + WEIGHT_REG * reg_mse + WEIGHT_NEG * neg_mse + WEIGHT_CAT * cat_loss
            imgs_bbs = detector.out_to_bbs(val_out, topk=100)
            for img_bbs in imgs_bbs:
                for img_bb in img_bbs:
                    coco_pred.dataset["annotations"].append(
                        {
                            "id": ann_id,
                            "bbox": [
                                img_bb["x"],
                                img_bb["y"],
                                img_bb["width"],
                                img_bb["height"],
                            ],
                            "area": img_bb["width"] * img_bb["height"],
                            "category_id": img_bb["category"],  # TODO replace with predicted category id
                            "score": img_bb["score"],
                            "image_id": image_id,
                        }
                    )
                    ann_id += 1
                image_id += 1
            count += len(val_img_batch) / BATCH_SIZE
        coco_pred.createIndex()
        coco_eval = COCOeval(val_dataloader.dataset.coco, coco_pred, iouType="bbox")
        coco_eval.params.useCats = 1  # TODO replace with 1 when categories are added
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        wandb.log(
            {
                "total val loss": (loss / count),
                "val loss pos": (total_pos_mse / count),
                "val loss neg": (total_neg_mse / count),
                "val loss reg": (total_reg_mse / count),
                "cat loss reg": (total_cat_loss / count),
                "val AP @IoU 0.5:0.95": coco_eval.stats[0],
                "val AP @IoU 0.5": coco_eval.stats[1],
                "val AR @IoU 0.5:0.95": coco_eval.stats[8],
            },
            step=current_iteration,
        )
        print(
            "Validation: {}, validation loss: {}".format(
                current_iteration, loss / count
            ),
        )
    detector.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    device.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    args = parser.parse_args()
    train(args.device)
