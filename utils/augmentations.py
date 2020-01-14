import torch
import torch.nn.functional as F
import numpy as np
import cv2
import viz
from rect import *
from viz import *


def rotate(img, targets, theta):
    '''
    Rotates an img (theta in degrees) and expands img to avoid cropping
    :param img: [H,W,(C)] matrix
    :param targets: [batch, class, x, y, w, h]
    :param theta: rotation parameters
    :return:
    '''
    h, w = img.shape[:2] # img shape has 3 dimensions
    img_center = (w/2, h/2) # getRotationimgrix2D needs coordinates in reverse order (width, height) compared to shape

    rot_mat = cv2.getRotationMatrix2D(img_center, theta, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0,0])
    abs_sin = abs(rot_mat[0,1])

    # find the new width and height bounds
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # subtract old img center (bringing img back to origo) and adding the new img center coordinates
    rot_mat[0, 2] += bound_w/2 - img_center[0]  #w
    rot_mat[1, 2] += bound_h/2 - img_center[1]  #h


    # rotate img with the new bounds and translated rotation imgrix
    rotated_img = cv2.warpAffine(img, rot_mat, (bound_w, bound_h))
    if len(rotated_img.shape)<3:
        rotated_img = np.expand_dims(rotated_img,axis=2)
    r_h, r_w = rotated_img.shape[:2]

    # rotate the targets
    t_x, t_y = targets[:,2].numpy(), targets[:,3].numpy()
    t_w, t_h = targets[:,4], targets[:,5]
    # rotate xy
    xy = PtsOnDstImg(np.column_stack((t_x * w,t_y * h)), rot_mat)

    #new x,y
    rotated_targets = targets.clone()
    rotated_targets[:, 2:4] = torch.from_numpy(xy)
    rotated_targets[:, 2] /= r_w
    rotated_targets[:, 3] /= r_h
    # new w,h
    rotated_targets[:, 4] = (t_h * h * abs_sin + t_w * w * abs_cos) /r_w
    rotated_targets[:, 5] = (t_h * h * abs_cos + t_w * w * abs_sin) /r_h

    #viz
    #plot_bbox(img, targets, 'tmp1')
    #plot_bbox(rotated_img, rotated_targets, 'tmp2')

    return rotated_img, rotated_targets

def perspective(img, targets, t):
    '''
    :param img:
    :param targets:
    :param t: transform [[0,0], [w,h]] into [[0,0], [w,h + w*t]] in (x,y) format
    :return:
    '''
    h, w = img.shape[0], img.shape[1]
    if t>0:
        input_pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        output_pts = np.float32([[0, 0], [0, h], [w, t*w], [w, h + w*t]])
    else:
        t=abs(t)
        input_pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        output_pts = np.float32([[0, w*t], [0, h + w*t], [w, 0], [w, h]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Apply the perspective transformation to the image
    p_w, p_h = w, int( h + abs(t) * w )
    warped_img = cv2.warpPerspective(img, M, (p_w, p_h), flags=cv2.INTER_LINEAR)
    if len(warped_img.shape)<3:
        warped_img = np.expand_dims(warped_img,axis=2)

    # warp the targets
    t_x, t_y = targets[:, 2].numpy(), targets[:, 3].numpy()
    t_h = targets[:, 5] * (1 +  t)
    # warp xy
    xy = PtsOnDstImg(np.column_stack((t_x * w, t_y * h)), M)

    #new x,y,h
    warped_targets = targets.clone()
    warped_targets[:, 2:4] = torch.from_numpy(xy)
    warped_targets[:, 2] /= p_w
    warped_targets[:, 3] /= p_h
    warped_targets[:, 5] = t_h

    # viz
    #plot_bbox(img, targets, 'tmp1')
    #plot_bbox(warped_img, warped_targets, 'tmp2')

    return warped_img, warped_targets