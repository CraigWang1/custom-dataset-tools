#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:17:00 2020

@author: craig
Script to help visualize the predictions of object detection models.
"""
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np

bboxes = [[55, 37, 55+20, 37+22]]
labels = ['gate']
bbox_scores= [100]

fig,ax = plt.subplots(1)
im = np.array(Image.open('/home/craig/gate_dataset_COCO/data/COCO/images/train2017/2403.png'))
ax.imshow(im)

for i in range(len(bboxes)):
    x1 = bboxes[i][0]
    y1 = bboxes[i][1]  #get the coords of bounding box
    x2 = bboxes[i][2]
    y2 = bboxes[i][3]
    
    width = abs(x2-x1)  #get the width and height of bounding box
    height = abs(y2-y1)
    
    ax.imshow(im)
    # Create a Rectangle patch
    rect = matplotlib.patches.Rectangle((x1, y1),width,height,linewidth=2,edgecolor='b',facecolor='none')
    
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    plt.text(x1, y1, labels[i], fontsize=20, color='r')
    plt.text(x1, y2, bbox_scores[i], fontsize=16, color='r')

plt.show()
