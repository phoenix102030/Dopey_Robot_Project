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

import utils
from detector import Detector
from classifier import Classifier

# Changeable stuff:
model_pt = "det_02-17_roadSigns_1hr.pt"
class_model_pt = "test_classifier.pt"
frame = "test_images/img_1.jpg"  # TODO: change to realsense feed

################################

detector = Detector() # Initialize the model
detector.load_state_dict(torch.load(model_pt))
detector.eval() # Set the model to evaluation mode

classifier = Classifier()  # Initialize the classifier
classifier.load_state_dict(torch.load(class_model_pt))  # Load the classifier weights
classifier.eval()  # Set the classifier to evaluation mode



image = Image.open(frame) # Load the image
original_image = copy.deepcopy(image) # Keep a copy of the original image for later

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image = transform(image)
image = image.unsqueeze(0) # Add an extra dimension for batch (PyTorch models expect a batch dimension)

with torch.no_grad():
    output = detector(image).cpu()
    bbs = detector.out_to_bbs(output, 0.5)

# Perform Non-Maximum Suppression
print("Bounding boxes before NMS:", len(bbs[0]))
bbs = utils.nms(bbs[0], iou_threshold=0.5)

# For each bounding box
for bb in bbs:
    # Extract the corresponding region from the original image
    region = original_image.crop((bb[0], bb[1], bb[2], bb[3]))
    
    # Transform the region (resize, normalize, etc.)
    region = transform(region)
    region = region.unsqueeze(0)  # Add an extra dimension for batch
    
    # Pass the region through the classifier
    with torch.no_grad():
        class_output = classifier(region)
    
    # Print the classifier output
    print("Classifier output:", class_output)


# Draw the bounding boxes on the image
result_image = utils.draw_detections(
    image.squeeze().permute(1, 2, 0),  # Remove batch dimension and convert to (height, width, channels)
    bbs,  # Use the bounding boxes after NMS
    channel_first=False,  # The image is now (height, width, channels)
)

# Print resulting image shape
print("imshape:", result_image.shape)
print("Bounding boxes:", bbs)

# Display the image
plt.imshow(result_image)
plt.show()