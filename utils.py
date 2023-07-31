import collections
import logging
import math
import os
from datetime import datetime

import dateutil.tz
import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2


def imread_CS_py(imgName):
    block_size =33
    Img = cv2.imread(imgName, 1)
    Img_rec_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Iorg = np.array(Image.open(imgName), dtype='float32')  # 读图
    if len(Iorg.shape) == 3: #rgb转y
        Iorg = test_rgb2ycbcr(Iorg)

    [row, col] = Iorg.shape  # 图像的 形状
    row_pad = block_size-np.mod(row,block_size)  # 求余数操作
    col_pad = block_size-np.mod(col,block_size)  # 求余数操作，用于判断需要补零的数量
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new,Img_rec_yuv]

def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape  # 当前图像的形状
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)  # 一共有多少个 模块
    img_col = np.zeros([block_size**2, block_num])  # 把每一块放进每一列中， 这就是容器
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            count = count + 1
    return img_col

def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def test_rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    rlt = rlt.round()

    return rlt.astype(in_img_type)


def img2patches(imgs,patch_size:tuple,stride_size:tuple):
    """
    Args:
        imgs: (H,W)/(H,W,C)/(B,H,W,C)
        patch_size: (patch_h, patch_w)
        stride_size: (stride_h, stride_w)
    """
    # print('imgs shape', imgs.shape)

    if imgs.ndim == 2:
        # (H,W) -> (1,H,W,1)
        imgs = np.expand_dims(imgs,axis=2)
        imgs = np.expand_dims(imgs,axis=0)
    elif imgs.ndim == 3:
        # (H,W,C) -> (1,H,W,C)
        imgs = np.expand_dims(imgs,axis=0)
    b,h,w,c = imgs.shape
    p_h,p_w = patch_size
    s_h,s_w = stride_size

    # assert (h-p_h) % s_h == 0 and (w-p_w) % s_w == 0

    n_patches_y = (h - p_h) // s_h + 1
    # print('n_patches_y',n_patches_y)
    n_patches_x = (w - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    n_patches = n_patches_per_img * b
    patches = np.empty((n_patches,p_h,p_w,c),dtype=imgs.dtype)

    patch_idx = 0
    for img in imgs:
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * s_h
                y2 = y1 + p_h
                x1 = j * s_w
                x2 = x1 + p_w
                patches[patch_idx] = img[y1:y2, x1:x2]
                patch_idx += 1
    return patches

def unpatch2d(patches, imsize: tuple, stride_size: tuple):
    '''
        patches: (n_patches, p_h, p_w,c)
        imsize: (img_h, img_w)
    '''
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    n_patches,p_h,p_w,c = patches.shape
    s_h, s_w = stride_size

    # assert (i_h - p_h) % s_h == 0 and (i_w - p_w) % s_w == 0

    n_patches_y = (i_h - p_h) // s_h + 1
    n_patches_x = (i_w - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    batch_size = n_patches // n_patches_per_img

    imgs = np.zeros((batch_size,i_h,i_w,c))
    weights = np.zeros_like(imgs)


    for img_idx, (img,weights) in enumerate(zip(imgs,weights)):
        start = img_idx * n_patches_per_img

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * s_h
                y2 = y1 + p_h
                x1 = j * s_w
                x2 = x1 + p_w
                patch_idx = start + i*n_patches_x+j
                img[y1:y2,x1:x2] += patches[patch_idx]
                weights[y1:y2, x1:x2] += 1
    imgs /= weights

    return imgs.astype(patches.dtype)
