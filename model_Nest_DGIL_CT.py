
import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
# from skimage.transform import radon, iradon
from torch_radon import Radon

###########################################################################
# Define GIL-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self, features):
        super(BasicBlock, self).__init__()

        self.Sp = nn.Softplus()
        self.weight1 = nn.Parameter(torch.Tensor([1]))
        self.weight2 = nn.Parameter(torch.Tensor([1]))
        self.weight3 = nn.Parameter(torch.Tensor([1]))
        self.weight4 = nn.Parameter(torch.Tensor([1]))

        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv5_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv5_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, yold, z, theta, sinogram, lambda_step, soft_thr):
        # rk block in the paper
        radon = Radon(yold.shape[2], theta, det_count=729)
        sino_pred = radon.forward(z)
        filtered_sinogram = radon.filter_sinogram(sino_pred)
        X_fbp = radon.backprojection(filtered_sinogram - sinogram)

        x_input = yold - self.Sp(lambda_step) * X_fbp

        x = self.conv_D(x_input)
        x_D = F.relu(x)
        x = self.conv1_forward(x_D)
        x_f1 = F.relu(x)
        x = self.conv2_forward(x_f1)
        x_f2 = F.relu(x)
        x = self.conv3_forward(x_f2)
        x_f3 = F.relu(x)
        x = self.conv4_forward(x_f3)
        x_f4 = F.relu(x)
        x_forward = self.conv5_forward(x_f4)
        ##########################################################################################
        # adaptive remainder
        # p=1
        x_st1 = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))
        # p=2
        x_st2 = x_forward / (1 + 2 * self.Sp(soft_thr))
        # p=0
        w = (self.Sp(soft_thr) * 2) ** (1 / 2)
        x_tmp = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - w))
        one = torch.ones_like(x_tmp)
        x_tmp1 = torch.where(torch.abs(x_tmp) > 0, one, x_tmp)
        x_st3 = x_tmp + x_tmp1 * w
        # p=3/2
        w4 = self.Sp(soft_thr)
        p4 = 1 + 16 * torch.abs(x_forward) / (9 * (w4 ** (2)))
        x_st4 = x_forward + 9 * (w4 ** (2)) * torch.sign(x_forward) * (1 - torch.sqrt(p4)) / 8

        # z in w_{k,n+1}
        x_st = (self.weight1 * x_st1 + self.weight2 * x_st2 + self.weight3 * x_st3 + self.weight4 * x_st4) / \
               (self.weight1 + self.weight2 + self.weight3 + self.weight4)
        # print('weight1=', self.weight1, 'weight2=', self.weight2, 'weight3=', self.weight3, 'weight4=', self.weight4)
        ############################################################################################

        x = self.conv1_backward(x_st)
        x_b1 = F.relu(x)
        x = self.conv2_backward(x_b1 + x_f4)
        x_b2 = F.relu(x)
        x = self.conv3_backward(x_b2 + x_f3)
        x_b3 = F.relu(x)
        x = self.conv4_backward(x_b3 + x_f2)
        x_b4 = F.relu(x)
        x = self.conv5_backward(x_b4 + x_f1)
        x_backward = F.relu(x)
        x_G = self.conv_G(x_backward + x_D)

        x_pred = F.relu(x_input + x_G)  # prediction output (skip connection); non-negative output

        return [x_pred, x_pred, x_pred]
#####################################################################################################
# Define ISTA-Net-plus
class GILNet(torch.nn.Module):
    def __init__(self, LayerNo, num_features):
        super(GILNet, self).__init__()
        self.LayerNo = LayerNo
        self.Sp = nn.Softplus()

        onelayer = []
        for i in range(LayerNo):
            onelayer.append(BasicBlock(num_features))

        self.fcs = nn.ModuleList(onelayer)

        # thresholding value
        self.w_theta1 = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta1 = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu1 = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu1 = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_gamma = nn.Parameter(torch.Tensor([0.5]))
        self.b_gamma = nn.Parameter(torch.Tensor([0]))

    # def forward(self, x0, sinogram, theta, theta_label):
    def forward(self, cond, x0, sinogram, theta, theta_label):
        x = x0
        xold = x
        yold = xold
        for i in range(self.LayerNo):
            theta1_ = self.w_theta1 * i + self.b_theta1  # parameter calculate
            mu1_ = self.w_mu1 * i + self.b_mu1
            gamma_ = self.Sp(self.w_gamma * i + self.b_gamma)
            x_coef = torch.tensor([1 - gamma_, gamma_])
            gamma_s = F.softmax(x_coef)  # ensure coef is positive
            z = gamma_s[0] * xold + gamma_s[1] * yold

            # [ynew, layer_sym] = self.fcs[i](yold, z, PhiTPhi, PhiTb, mu1_, theta1_)
            [ynew, layer_sym, layer_st] = self.fcs[i](yold, z, theta, sinogram, mu1_, theta1_)

            xnew = gamma_s[0] * xold + gamma_s[1] * ynew
            xold = xnew
            yold = ynew

        x_final = xnew
        return [x_final, x_final, x_final]
