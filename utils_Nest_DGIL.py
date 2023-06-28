import torch
import numpy as np
import math
import copy
from torch.utils.data import Dataset

def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)
########################################################################################
def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape
    return [Iorg, row, col, Ipad, row_new, col_new]
########################################################################################
def imread_CS_py_overlap(Iorg, overlap_size):
    block_size = 33
    block_stride = block_size-overlap_size
    [row, col] = Iorg.shape
    row_pad = block_stride-np.mod(row-block_size,block_stride)
    col_pad = block_stride-np.mod(col-block_size,block_stride)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape
    return [Iorg, row, col, Ipad, row_new, col_new]
########################################################################################
def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col
########################################################################################
def img2col_py_overlap(Ipad, block_size,overlap_size):
    block_stride = block_size-overlap_size
    [row, col] = Ipad.shape
    row_block = (row-block_size)/block_stride+1
    col_block = (col-block_size)/block_stride+1
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_stride):
        for y in range(0, col-block_size+1, block_stride):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            count = count + 1
    return img_col
######################################################################################
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
######################################################################################
def col2im_CS_py_overlap(X_col, row, col, row_new, col_new,overlap_size):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    block_stride = block_size-overlap_size
    gap_single = int(overlap_size/2)
    count = 0
    for x in range(0, row_new-block_size+1, block_stride):
        for y in range(0, col_new-block_size+1, block_stride):
            x_pad = X_col[:, count].reshape([block_size, block_size])
            if x==0 and y==0:  # original block
                X0_rec[0:block_size-gap_single, 0:block_size-gap_single] \
                    = x_pad[:block_size-gap_single,:block_size-gap_single]
            elif x==0 and y!=0 and y!=(col_new-block_size):  # coboundary
                X0_rec[:block_size-gap_single, y+gap_single:y+block_size-gap_single] \
                    = x_pad[:block_size-gap_single,gap_single:block_size-gap_single]
            elif x == 0 and y == (col_new - block_size):  # corner of coboundary
                X0_rec[:block_size-gap_single, y :y + block_size] = x_pad[:block_size-gap_single, :block_size]
            elif x!=0 and y==0 and x!=(row_new-block_size):   # left boundary
                X0_rec[x+gap_single:x+block_size-gap_single, :block_size-gap_single] \
                    = x_pad[gap_single:block_size-gap_single,:block_size-gap_single]
            elif y==0 and x==(row_new-block_size):   # corner of boundary
                X0_rec[x:x+block_size, :block_size-gap_single] = x_pad[:block_size,:block_size-gap_single]
            elif x == (row_new-block_size) and  y!=0 and y!=(col_new-block_size):   # bottom boundary
                X0_rec[x:x+block_size, y+gap_single:y + block_size-gap_single]\
                    = x_pad[:block_size, gap_single:block_size-gap_single]
            elif y == (col_new-block_size) and  x!=0 and x!=(row_new-block_size):   # right boundary
                X0_rec[x+gap_single:x + block_size-gap_single, y:y + block_size] \
                    = x_pad[gap_single:block_size-gap_single, :block_size]
            elif x == (row_new-block_size) and y == (col_new-block_size): # low right corner
                X0_rec[x :x + block_size, y:y + block_size] = x_pad[:block_size, :block_size]
            else:  # center
                X0_rec[x+gap_single:x+ gap_single+block_stride, y+gap_single:y + gap_single+block_stride] = \
                    x_pad[gap_single:gap_single+ block_stride, gap_single:gap_single+ block_stride]
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec
#####################################################################################
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

