#!/usr/bin/env python

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from PL_development.dev import Transformer as tf
from utils.PseudoLiDAR import PseudoLiDAR
from utils.model import BtsModel

def _load_state(model):
    """The _load_state method loads the weights from a state_dict into the model
    and returns it.

    Args: model
        
    Returns: model
    """
    # model init
    model = nn.DataParallel(model)

    # original saved file with DataParallel
    checkpoint_path = "./utils/model_weights"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) # add map_location=torch.device
            
    # load params
    model.load_state_dict(checkpoint['model'])

    return model

#  import image
image = cv2.imread("./data/rgb.png")

# pass image through depth network 
# to get depth image.
transform_pipeline = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((352, 1216)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
                                                )

imgData = transform_pipeline(image)
imgData = imgData.unsqueeze(0)

#  load model
model = _load_state(BtsModel())

#  forward pass
out = model(imgData, 0)

# image should be (height x width x channels)
# transformed image is test_img
output = out[4][0].cpu().detach().numpy()
output = np.transpose(output, (1, 2, 0))
output = np.squeeze(output).astype(np.float32) 
print("depth image processed")

# convert depth to PL point cloud
CALIB = "./data/"
PL = PseudoLiDAR(CALIB, 10)

cloud = PL.project_PL(output)
print("could processed")
# (123577, 4) -> real velo (all points)
# (234995, 4) -> our velo  (only fov, sparsity = 0)

# create variable cloud output
#  whilst processing: DONE in PseudoLiDAR.py
print(cloud.shape)

# test accuracy
# compare new cloud with velo cloud
def plot(img):
    # display result image
    fig=plt.figure(figsize=(13, 3))
    fig.add_subplot(2, 1, 1)
    plt.imshow(img, cmap='gist_gray')
    plt.title("Projected points ")

    fig.add_subplot(2, 1, 2)
    plt.imshow(img, cmap='gist_gray', interpolation='nearest')
    plt.title("Projected points (interpolated)")
    plt.show()


# using pillow
img = np.asarray(Image.open("./data/raw_vel.png"))
img = img/256.0
# plot(img)

TF = tf("./data/")
depth_from_cloud = TF.test_3d_to_2d(cloud)
print(depth_from_cloud.shape)
plot(depth_from_cloud)

# 2. remove all cpu processing 
# 3. add focal






