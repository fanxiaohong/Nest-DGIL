import torch
import torch.nn as nn
import numpy as np
import math

#########################################################################################
# # MRI inverse transform F
# class FFT_Mask_ForBack(torch.nn.Module):
#     def __init__(self):
#         super(FFT_Mask_ForBack, self).__init__()
#
#     def forward(self, x, mask):
#         x_dim_0 = x.shape[0]
#         x_dim_1 = x.shape[1]
#         x_dim_2 = x.shape[2]
#         x_dim_3 = x.shape[3]
#         x = x.view(-1, x_dim_2, x_dim_3, 1)
#         y = torch.zeros_like(x)
#         z = torch.cat([x, y], 3)
#         fftz = torch.fft(z, 2)
#         z_hat = torch.ifft(fftz * mask, 2)
#         x = z_hat[:, :, :, 0:1]
#         x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
#         return x
##########################################################################
def psnr(img1, img2):
    # img1.astype(np.float32)
    # img2.astype(np.float32)
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
##########################################################################
def compute_measure(y_gt, y_pred, data_range):
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_rmse)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))
##########################################################################

