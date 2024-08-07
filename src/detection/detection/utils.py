"""Utility functions to handle object detection."""
from typing import Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .detector import BoundingBox

import yaml
import cv2

def decode_calibration_yaml(yaml_file):
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        calibration_data = yaml.safe_load(file)

    # Extract the necessary information
    image_width = calibration_data['image_width']
    image_height = calibration_data['image_height']
    camera_matrix = np.array(calibration_data['camera_matrix']['data']).reshape(calibration_data['camera_matrix']['rows'], calibration_data['camera_matrix']['cols'])
    distortion_coefficients = np.array(calibration_data['distortion_coefficients']['data']).reshape(calibration_data['distortion_coefficients']['rows'], calibration_data['distortion_coefficients']['cols'])


    return image_width, image_height, camera_matrix, distortion_coefficients


def get_label_names():
    return [
        {"id": 0, "name": "binky"},
        {"id": 1, "name": "hugo"},
        {"id": 2, "name": "slush"},
        {"id": 3, "name": "muddles"},
        {"id": 4, "name": "kiki"},
        {"id": 5, "name": "oakie"},
        {"id": 6, "name": "cube"},
        {"id": 7, "name": "sphere"}
    ]

def get_label_names_B():
    return [
        {"id": 0, "name": "none"},
        {"id": 1, "name": "blue cube"},
        {"id": 2, "name": "binky"},
        {"id": 3, "name": "box"},
        {"id": 4, "name": "blue sphere"},
        {"id": 5, "name": "green cube"},
        {"id": 6, "name": "green sphere"},
        {"id": 7, "name": "hugo"},
        {"id": 8, "name": "kiki"},
        {"id": 9, "name": "muddles"},
        {"id": 10, "name": "oakie"},
        {"id": 11, "name": "red cube"},
        {"id": 12, "name": "red sphere"},
        {"id": 13, "name": "slush"},
        {"id": 14, "name": "wc"},
    ]



def draw_detections(
    image: Image,
    bbs: List[BoundingBox],
    category_dict: Optional[Dict[int, str]] = None,
    confidence: Optional[torch.Tensor] = None,
    channel_first: bool = False,
) -> torch.Tensor:
    """Add bounding boxes to image.

    Args:
        image:
            The image without bounding boxes.
        bbs:
            List of bounding boxes to display.
            Each bounding box dict has the format as specified in
            detector.Detector.decode_output.
        category_dict:
            Map from category id to string to label bounding boxes.
            No labels if None.
        channel_first:
            Whether the returned image should have the channel dimension first.

    Returns:
        The image with bounding boxes. Shape (H, W, C) if channel_first is False,
        else (C, H, W).
    """
    fig, ax = plt.subplots(1)
    plt.imshow(image)
    if confidence is not None:
        plt.imshow(
            confidence,
            interpolation="nearest",
            extent=(0, 640, 480, 0),
            alpha=0.5,
        )
    for bb in bbs:
        rect = patches.Rectangle(
            (bb["x"], bb["y"]),
            bb["width"],
            bb["height"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        if category_dict is not None:
            plt.text(
                bb["x"],
                bb["y"],
                category_dict[bb["category"]]["name"],
            )


    # Save matplotlib figure to numpy array without any borders
    plt.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.canvas.draw()
    data = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    if channel_first:
        data = data.transpose((2, 0, 1))  # HWC -> CHW

    return torch.from_numpy(data).float() / 255


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


def IoU(bb1, bb2):
    """
    Martin,

    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bb1, bb2: The bounding boxes to compare. Each bounding box is a dictionary containing 'x', 'y', 'width', and 'height'.
        
    Returns:
        The IoU of bb1 and bb2.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(bb1['x'], bb2['x'])
    y1 = max(bb1['y'], bb2['y'])
    x2 = min(bb1['x'] + bb1['width'], bb2['x'] + bb2['width'])
    y2 = min(bb1['y'] + bb1['height'], bb2['y'] + bb2['height'])
    
    # Calculate the area of the intersection rectangle
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of each bounding box
    bb1_area = bb1['width'] * bb1['height']
    bb2_area = bb2['width'] * bb2['height']
    
    # Calculate the IoU
    iou = intersection / (bb1_area + bb2_area - intersection)
    
    return iou

def nms(bbs, iou_threshold):
    """
    Martin,
    
    Perform Non-Maximum Suppression, adjusted for first NN (i.e. detection_NN.py).
    
    Args:
        bbs: List of bounding boxes. Each bounding box is a dictionary containing 'x', 'y', 'width', 'height', and 'score'.
        iou_threshold: IoU threshold for suppression. Bounding boxes with IoU greater than this threshold will be suppressed.
        
    Returns:
        List of bounding boxes after NMS.
    """
    # Sort the bounding boxes by score in descending order
    bbs = sorted(bbs, key=lambda x: x['score'], reverse=True)
    
    # List to hold the final bounding boxes after NMS
    final_bbs = []
    
    while bbs:
        # Take the bounding box with the highest score
        bb = bbs.pop(0)
        
        # Add it to the final list
        final_bbs.append(bb)
        
        # Compare this bounding box with the rest
        bbs = [other_bb for other_bb in bbs if IoU(bb, other_bb) < iou_threshold]
        
    return final_bbs




def ros_to_pil(msg):
    import io
    from PIL import Image

    try:
        pil_image = Image.open(io.BytesIO(bytearray(msg.data)))
    except Exception as e:
        print(f"Error converting ROS Image to PIL Image: {e}")
        # Return a default image in case of error
        pil_image = Image.new('RGB', (640, 480))  # Replace with your default image

    return pil_image

