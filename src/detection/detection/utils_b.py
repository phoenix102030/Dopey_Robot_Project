"""Utility functions to handle object detection."""

try:

    from .detector_b import BoundingBox
except ImportError:
    from detector_b import BoundingBox
    
from typing import Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from cv_bridge import CvBridge
import json
from torchvision import transforms


import io
import cv2
import time

bridge = CvBridge()


def draw_detections(
    image: Image,
    bbs: List[BoundingBox],
    #category_dict: Optional[Dict[int, str]] = None,
    confidence: Optional[torch.Tensor] = None,
    channel_first: bool = False,
    centroids: Optional[List[tuple]] = None
) -> torch.Tensor:
    """Add bounding boxes to image.

    Args:
        image:
            The image without bounding boxes.
        bbs:
            List of bounding boxes to display.
            Each bounding box dict has the format as specified in
            detector.Detector.decode_output.
        category_dict:out_to_bbs
            Map from category id to string to label bounding boxes.
            No labels if None.
        channel_first:
            Whether the returned image should have the channel dimension first.

    Returns:
        The image with bounding boxes. Shape (H, W, C) if channel_first is False,
        else (C, H, W).
    """

    try:
        with open("./dd2419_23_data/dd2419_23_data_b/annotations/training.json", 'r') as file:
            data = json.load(file)
            print(data['categories'][0])  # Test accessing the first category
    except Exception as e:
        print(f"Error: {e}")
    
    categories = data['categories']
    
    category_dict = {category['id']: {'name': category['name']} for category in categories}




    # np_img = np.array(image)
    # for bb in bbs:
    #     start_rect = (int(bb["x"]), int(bb["y"]))
    #     end_rect = (int(bb["width"] + bb["x"]), int(bb["height"] + bb["y"]))
    #     cv2.rectangle(np_img, start_rect, end_rect, color=(255,0,0), thickness=2)

    fig, ax = plt.subplots(1)
    plt.imshow(image)
    # if confidence is not None:
    #     plt.imshow(
    #         confidence,
    #         interpolation="nearest",
    #         extent=(0, 640, 480, 0),
    #         alpha=0.5,
    #     )

    for i, bb in enumerate(bbs):
        rect = patches.Rectangle(
            (bb["x"], bb["y"]),
            bb["width"],
            bb["height"],
            linewidth=1,
            edgecolor="r" if bb["score"] < 1.0 else 'g',
            facecolor="none",
        )
        # circ = patches.Circle(centroids[i], 3, edgecolor="r")
        # ax.add_patch(circ)

        ax.add_patch(rect)
        if centroids is not None:
            circ = patches.Circle(centroids[i], 3, edgecolor="r")
            ax.add_patch(circ)
            
        if category_dict is not None:
            plt.text(
                bb["x"],
                bb["y"],
                category_dict[bb["category"]]["name"],
                color = 'red',
            )
        offset = len(category_dict) * 2

        plt.text(
            bb["x"] + offset,
            bb["y"],
            "%.2f" % bb["score"],
            color = 'red',
        )

    # Save matplotlib figure to numpy array without any borders
    plt.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Convert RGB to BGR for OpenCV
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    ros_image = bridge.cv2_to_imgmsg(data, "bgr8")
    # plt.close(fig)
    plt.close()

    return ros_image
    # return torch.from_numpy(data).float() / 255

def nms_b(bbs_list, iou_thresh):
    """ Perform Non-Maximum Suppression on a list of bounding boxes.
    
    Martin
    
    Note: adjusted for "b" dataset i.e. detection_NN_b.py
    """
    nms_bbs_list = []
    for bbs in bbs_list:
        # Sort bounding boxes by score
        bbs = sorted(bbs, key=lambda x: x['score'], reverse=True)

        # Perform Non-Maximum Suppression
        nms_bbs = []
        while bbs:
            # Take the bounding box with the highest score
            best_bb = bbs.pop(0)

            nms_bbs.append(best_bb)

            # Compare the best bounding box with the rest
            bbs = [bb for bb in bbs if iou_b(best_bb, bb) < iou_thresh]

        nms_bbs_list.append(nms_bbs)

    return nms_bbs_list


def iou_b(bb1, bb2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes

    intersect_w = _interval_overlap_b([bb1['x'], bb1['x']+bb1['width']], [bb2['x'], bb2['x']+bb2['width']])
    intersect_h = _interval_overlap_b([bb1['y'], bb1['y']+bb1['height']], [bb2['y'], bb2['y']+bb2['height']])

    intersect = intersect_w * intersect_h

    w1, h1 = bb1['width'], bb1['height']
    w2, h2 = bb2['width'], bb2['height']

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union

def _interval_overlap_b(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3



def save_model(model: torch.nn.Module, path: str) -> None:
    """Save model to disk.

    Args:
        model: The model to save.
        path: The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    """Load model weights from disk.

    Args:
        model: The model to load the weights into.
        path: The path from which to load the model weights.
        device: The device the model weights should be on.

    Returns:
        The loaded model (note that this is the same object as the passed model).
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
