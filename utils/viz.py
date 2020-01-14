from __future__ import division

import cv2
import os
import sys
import time
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from utils.utils import load_classes

def plot_bbox(img, targets, filename = 'tmp', format = 'png'):
    '''
    :param img: H*W
    :param targets: batch, cls, x, y, w, h
    :param filename:
    :param format:
    :return:
    '''

    img_new = img
    if np.max(img_new) <= 1:
        img_new *= 255
    if len(img_new.shape) == 2 or img_new.shape[-1] == 1:
        img_new = cv2.cvtColor(img_new,cv2.COLOR_GRAY2BGR)
    H,W,_ = img_new.shape
    # Draw bounding boxes and labels of detections
    if targets is not None:
        # Rescale boxes to original image
        #import pdb; pdb.set_trace()
        boxes = []
        #for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        for _, _, x, y, w, h in targets.numpy():
            x*=W
            w*=W
            y*=H
            h*=H
            box = [[int(x-w/2),int(y-h/2)],
                   [int(x+w/2),int(y-h/2)],
                   [int(x+w/2),int(y+h/2)],
                   [int(x-w/2),int(y+h/2)]]
            boxes.append(box)
            print(x,y,w,h)
            print(box)
        boxes = np.array(boxes)
        cv2.drawContours(img_new, boxes, -1, (0, 0, 255), 2)
    # Save generated image with detections
    cv2.imwrite(os.path.join('tmp',filename+'.'+format),img_new)