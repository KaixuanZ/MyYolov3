import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd

from utils.augmentations import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms

'''
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad
'''

def resize(image, size):
    img_size=[int(size[0]/32)*32,int(size[1]/32)*32]
    image = F.interpolate(image.unsqueeze(0), size=img_size, mode="nearest").squeeze(0)
    return image

'''
def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images
'''

class DetectionResults():
    def __init__(self, datas = None, filenames = None, classes = [], outputpath = ''):
        self.filenames = filenames
        self.datas = datas  # [ [ [x1,y1,x2,y2,confidence, number_of_classes] ] ]
        self.classes = classes
        self.outputpath = outputpath
        self.header = ['x1','y1','x2','y2','confidence'] + classes

    def to_df(self,data):
        return pd.DataFrame(data.numpy(), columns=self.header)

    def to_csvs(self):
        for i,data in enumerate(self.datas):
            filename = self.filenames[i].split('/')[-1].split('.')[0] + '.csv'
            self.to_df(data).to_csv(os.path.join(self.outputpath,filename),index=False)
            print("data saved to "+os.path.join(self.outputpath,filename))

    def from_csvs(self):
        pass

    def to_jsons(self):
        pass

    def from_jsons(self):
        pass

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('L'))
        # Pad to square resolution
        #img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.factor = 1
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('L'))
        #print(img.shape)
        # Handle images with less than three channels
        if len(img.shape) != 3:	#only H,W, add another dimension C
            img = img.unsqueeze(0) # C=1
            #img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        #img, pad = pad_to_square(img, 0)
        img = resize(img, self.img_size)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) #(x, y, w, h)
            '''
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)
            
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            '''
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            # we can save data in this class but if the training data is too much we cannot remember them in memory
            # add noise

            img, targets = rotate(img, targets, 1)
            if np.random.random() < 0.5:
                # random small affine transform
                pass
            else:
                # random rotation (small degree)
                pass

        return img_path, img, targets

    def collate_fn(self, batch):
        #collate list of samples into batches
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.factor = random.random()*0.2+0.9	#[0.9,1.1]
        img_size = [int(self.img_size[0]*self.factor), int(self.img_size[1]*self.factor)]
        # Resize images to input shape
        imgs = torch.stack([resize(img, img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
