"""Baseline detector model.

Inspired by
You only look once: Unified, real-time object detection, Redmon, 2016.
"""
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms
from torchvision.ops import nms
import albumentations as A
import random

class BoundingBox(TypedDict):
    """Bounding box dictionary.

    Attributes:
        x: Top-left corner column
        y: Top-left corner row
        width: Width of bounding box in pixel
        height: Height of bounding box in pixel
        score: Confidence score of bounding box.
        category: Category (not implemented yet!)
    """

    x: int
    y: int
    width: int
    height: int
    score: float
    category: int


class Detector(nn.Module):
    """Baseline module for object detection."""

    def __init__(self) -> None:
        """Create the module.

        Define all trainable layers.
        """
        super(Detector, self).__init__()

        self.features = models.mobilenet_v2(pretrained=True).features
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images

        #self.head = nn.Conv2d(in_channels=1280, out_channels=6, kernel_size=1)
        self.head = nn.Conv2d(in_channels=1280, out_channels=13, kernel_size=1)
        # 1x1 Convolution to reduce channels to out_channels without changing H and W

        # 1280x15x20 -> 5x15x20, where each element 5 channel tuple corresponds to
        #   (rel_x_offset, rel_y_offset, rel_x_width, rel_y_height, confidence
        # Where rel_x_offset, rel_y_offset is relative offset from cell_center
        # Where rel_x_width, rel_y_width is relative to image size
        # Where confidence is predicted IOU * probability of object center in this cell
        self.out_cells_x = 20
        self.out_cells_y = 15
        self.img_height = 480 #720.0 
        self.img_width =  640 #1280.0 

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Compute output of neural network from input.

        Args:
            inp: The input images. Shape (N, 3, H, W).

        Returns:
            The output tensor encoding the predicted bounding boxes.
            Shape (N, 5, self.out_cells_y, self.out_cells_y).
        """
        features = self.features(inp)
        out = self.head(features)  # Linear (i.e., no) activation

        return out

    

    def decode_output(
        self, out: torch.Tensor, threshold: Optional[float] = None, topk: int = 100
    ) -> List[List[BoundingBox]]:
        """Convert output to list of bounding boxes.

        Args:
            out (torch.tensor):
                The output tensor encoding the predicted bounding boxes.
                Shape (N, 5, self.out_cells_y, self.out_cells_y).
                The 5 channels encode in order:
                    - the x offset,
                    - the y offset,
                    - the width,
                    - the height,
                    - the confidence.
            threshold:
                The confidence threshold above which a bounding box will be accepted.
                If None, the topk bounding boxes will be returned.
            topk (int):
                Number of returned bounding boxes if threshold is None.

        Returns:
            List containing N lists of detected bounding boxes in the respective images.
        """
        bbs = []
        out = out.cpu()
        # decode bounding boxes for each image
        for o in out:
            img_bbs = []

            # find cells with bounding box center
            if threshold is not None:
                bb_indices = torch.nonzero(o[4, :, :] >= threshold)
            else:
                _, flattened_indices = torch.topk(o[4, :, :].flatten(), topk)
                bb_indices = np.array(
                    np.unravel_index(flattened_indices.numpy(), o[4, :, :].shape)
                ).T

            boxes = []
            scores = []

            # loop over all cells with bounding box center
            for bb_index in bb_indices:
                
                bb_coeffs = o[0:4, bb_index[0], bb_index[1]]
                bb_category = torch.argmax(o[5:, bb_index[0], bb_index[1]]).item()
                
                # decode bounding box size and position
                width = self.img_width * abs(bb_coeffs[2].item())
                height = self.img_height * abs(bb_coeffs[3].item())
                y = (
                    self.img_height / self.out_cells_y * (bb_index[0] + bb_coeffs[1])
                    - height / 2.0
                ).item()
                x = (
                    self.img_width / self.out_cells_x * (bb_index[1] + bb_coeffs[0])
                    - width / 2.0
                ).item()

                img_bbs.append(
                    {
                        "width": width,
                        "height": height,
                        "x": x,
                        "y": y,
                        "score": o[4, bb_index[0], bb_index[1]].item(),
                        "category": bb_category,
                    }
                )
                boxes.append([x,y,x+width, y+height])
                scores.append(o[4, bb_index[0], bb_index[1]])

            scores_tensor = torch.tensor(scores)
            boxes_tensor = torch.tensor(boxes)
            
            if len(boxes_tensor.shape) == 2:
                nms_bbs = nms(boxes=boxes_tensor, scores=scores_tensor, iou_threshold=0.5)
                img_bbs = [img_bbs[i] for i in nms_bbs]
            bbs.append(img_bbs)


        return bbs




    def input_transform(self, image: Image, anns: List) -> Tuple[torch.Tensor]:
        """Prepare image and targets on loading.

        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.

        Args:
            image:
                The image loaded from the dataset.
            anns:
                List of annotations in COCO format.

        Returns:
            Tuple:
                image: The image. Shape (3, H, W).
                target:
                    The network target encoding the bounding box.
                    Shape (5, self.out_cells_y, self.out_cells_x).
        """
    
        

        target = torch.zeros(13, self.out_cells_y, self.out_cells_x)
        if len(anns)>0:
            boxes = []
            class_labels = []
            for ann in anns:
                boxes.append([[ann["bbox"][0],ann["bbox"][1],ann["bbox"][2],ann["bbox"][3]]])
                class_labels.append([ann["category_id"]])

           
            #image augmentation 
            transform = A.Compose([
                    A.GaussNoise(p=0.2),
                    A.MotionBlur(p=0.8),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3),
                        A.Blur(blur_limit=3),
                    ],p=0.2),
                    A. OneOf([
                        A.CLAHE(clip_limit=2, p=0.2),
                        A.Sharpen(p=0.2),
                        A.RandomBrightnessContrast(p = 0.6),
                    ],p=0.2),
                    A.Affine(shear = {'x': random.randint(-15, 15), 'y': random.randint(-15, 15)},p=0.2)
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']), p=0.5)

            image = np.array(image)
            transformed = transform(image=image, bboxes=boxes[0], class_labels=class_labels)
            boxes = [transformed['bboxes']]
            image = Image.fromarray(transformed['image'], 'RGB')

            boxes = torch.Tensor(list(boxes))

            # Resize image and bounding box 
            image, boxes = self.resize(image, boxes, (self.img_height, self.img_width),False)

            
            for i in range(len(anns)):
                x = boxes[i][0][0]
                y = boxes[i][0][1]
                width = boxes[i][0][2]
                height = boxes[i][0][3]
                shape = [(x, y), (x +width, y+height)]
                # image1 = ImageDraw.Draw(image) 
                # image1.rectangle(shape, outline ="red")
                # image.show()

                x_center = x + width / 2.0
                y_center = y + height / 2.0
                x_center_rel = x_center / self.img_width * self.out_cells_x
                y_center_rel = y_center / self.img_height * self.out_cells_y
                x_ind = int(x_center_rel)
                y_ind = int(y_center_rel)
                x_cell_pos = x_center_rel - x_ind
                y_cell_pos = y_center_rel - y_ind
                rel_width = width / self.img_width
                rel_height = height / self.img_height

                # channels, rows (y cells), cols (x cells)
                target[4, y_ind, x_ind] = 1

                # bb size
                target[0, y_ind, x_ind] = x_cell_pos
                target[1, y_ind, x_ind] = y_cell_pos
                target[2, y_ind, x_ind] = rel_width
                target[3, y_ind, x_ind] = rel_height

                # category 
                category = ann["category_id"] + 5
                target[category,y_ind, x_ind] = 1
        else:
            image = transforms.functional.resize(image, (self.img_height, self.img_width))
        
        # Convert PIL.Image to torch.Tensor
        image = transforms.ToTensor()(image)

        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image)

        return image, target


    def input_transform_validation(self, image: Image, anns: List) -> Tuple[torch.Tensor]:
        """Prepare image and targets on loading.

        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.

        Args:
            image:
                The image loaded from the dataset.
            anns:
                List of annotations in COCO format.

        Returns:
            Tuple:
                image: The image. Shape (3, H, W).
                target:
                    The network target encoding the bounding box.
                    Shape (5, self.out_cells_y, self.out_cells_x).
        """
    
        

        target = torch.zeros(13, self.out_cells_y, self.out_cells_x)
        if len(anns)>0:
            boxes = []
            for ann in anns:
                boxes.append([[ann["bbox"][0],ann["bbox"][1],ann["bbox"][2],ann["bbox"][3]]])

            boxes = torch.Tensor(list(boxes)
                                 )
            # Resize image and bounding box 
            image, boxes = self.resize(image, boxes, (self.img_height, self.img_width),False)


            for i in range(len(anns)):
                x = boxes[i][0][0]
                y = boxes[i][0][1]
                width = boxes[i][0][2]
                height = boxes[i][0][3]
                shape = [(x, y), (x +width, y+height)]

                x_center = x + width / 2.0
                y_center = y + height / 2.0
                x_center_rel = x_center / self.img_width * self.out_cells_x
                y_center_rel = y_center / self.img_height * self.out_cells_y
                x_ind = int(x_center_rel)
                y_ind = int(y_center_rel)
                x_cell_pos = x_center_rel - x_ind
                y_cell_pos = y_center_rel - y_ind
                rel_width = width / self.img_width
                rel_height = height / self.img_height

                # channels, rows (y cells), cols (x cells)
                target[4, y_ind, x_ind] = 1

                # bb size
                target[0, y_ind, x_ind] = x_cell_pos
                target[1, y_ind, x_ind] = y_cell_pos
                target[2, y_ind, x_ind] = rel_width
                target[3, y_ind, x_ind] = rel_height

                # category 
                category = ann["category_id"] + 5
                target[category,y_ind, x_ind] = 1
                if category >= target.size(0):
                    print(f"Error: category index {category} is out of bounds.")
                    continue

        else:
            image = transforms.functional.resize(image, (self.img_height, self.img_width))
        
        # Convert PIL.Image to torch.Tensor
        image = transforms.ToTensor()(image)

        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image)

        return image, target


    def resize(self, image, boxes, dims=(480, 640), return_percent_coords=True):
        """
        Resize image. 
        Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
        you may choose to retain them.
        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
        """
        # Resize image
        new_image = transforms.functional.resize(image, dims)
        if boxes.size()[0]>0:
            # Resize bounding boxes
            old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)

            new_boxes = boxes / old_dims  # percent coordinates

            if not return_percent_coords:
                new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
                new_boxes = new_boxes * new_dims

            return new_image, new_boxes

        return new_image