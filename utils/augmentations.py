import torch
import torch.nn.functional as F
import numpy as np

'''
def horisontal_flip(image, targets):
    image = torch.flip(image, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return image, targets
'''

def rotate(img, targets, theta):
    """
    Rotates an img (theta in degrees) and expands img to avoid cropping
    """

    height, width = img.shape[:2] # img shape has 3 dimensions
    img_center = (width/2, height/2) # getRotationimgrix2D needs coordinates in reverse order (width, height) compared to shape

    rot_mat = cv2.getRotationMatrix2D(img_center, theta, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0,0])
    abs_sin = abs(rot_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old img center (bringing img back to origo) and adding the new img center coordinates
    rot_mat[0, 2] += bound_w/2 - img_center[0]
    rot_mat[1, 2] += bound_h/2 - img_center[1]

    # rotate img with the new bounds and translated rotation imgrix
    rotated_img = cv2.warpAffine(img, rot_mat, (bound_w, bound_h))
    import pdb;
    pdb.set_trace()

    #rotate the targets
    xywhs = targets[:,1:]
    # transform xywhr into pts and rotate
    pts = xywhs2pts(xywhs)
    pts = PtsOnDstImg(pts, rot_mat)
    # pts to rect (xywh) and put it back
    xywhs = torch.from_numpy(pts2xywhs(pts))
    rotated_targets = targets
    rotated_targets[:,1:] = xywhs

    return rotated_img, rotated_targets

def affine(image, targets):
    return image, targets