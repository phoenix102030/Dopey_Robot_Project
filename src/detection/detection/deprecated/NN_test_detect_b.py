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

import utils_b
from detector_b import Detector
import cv2
from cv_bridge import CvBridge
import numpy as np

# Changeable stuff:
model_pt = "model_B.pt"
frame = "./dd2419_23_data/dd2419_23_data_a/test_images/box13.jpg"  # TODO: change to realsense feed


IMSIZE_X = 640
IMSIZE_Y = 480
crop_h = 720
crop_w = int(crop_h * 4 / 3)
################################

detector = Detector() # Initialize the model
detector.load_state_dict(torch.load(model_pt))
detector.eval() # Set the model to evaluation mode


image = Image.open(frame) # Load the image
original_image = copy.deepcopy(image) # Keep a copy of the original image for later

transform = v2.Compose(
    [
        v2.CenterCrop((crop_h, crop_w)), # crop into 4:3 ratio which 640x480 is
        v2.Resize((IMSIZE_Y, IMSIZE_X)), # orig is 1280x720 but we will feed realsense in 640x480
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image = transform(image)
image = image.unsqueeze(0) # Add an extra dimension for batch (PyTorch models expect a batch dimension)

with torch.no_grad():
    output = detector(image).cpu()
    #bbs = detector.decode_output(output, 0.7) #model A
    bbs = detector.out_to_bbs(output, 0.7) #model B
# Perform Non-Maximum Suppression
# print("Bounding boxes before NMS:", len(bbs[0]))
# bbs = utils.nms(bbs[0], iou_threshold=0.2) # lower thresh = less boxes (0->1)
for bb_list in bbs:  # Iterate through the outer list (which might just be a single-element list in your case)
    for bb in bb_list:  # Iterate through the list of bounding box dictionaries
        print(bb['category']) 

# print ("bbs",bbs)




# Draw the bounding boxes on the image
result_image = utils_b.draw_detections(
    image.squeeze().permute(1, 2, 0),  # Remove batch dimension and convert to (height, width, channels)
    bbs[0],  # Use the bounding boxes after NMS
    channel_first=False,  # The image is now (height, width, channels)
)

bridge = CvBridge()

# Convert ROS image message to OpenCV image
cv_image = bridge.imgmsg_to_cv2(result_image, desired_encoding='passthrough')

# Convert the image to a format suitable for matplotlib (if needed)
if cv_image.dtype != np.uint8:
    cv_image = (cv_image * 255).astype(np.uint8)

# Print resulting image shape
#print("imshape:", result_image.shape)
print("Bounding boxes:", bbs)
rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


# Display the image
#plt.imshow(result_image)
plt.imshow(rgb_image)  # Convert BGR to RGB for matplotlib
plt.show()