import os
from datetime import datetime
import cv2
import glob
import torch
import platform
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn as nn
import scipy.io as sio
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim as ssim
from utils_Nest_DGIL import imread_CS_py,imread_CS_py_overlap, img2col_py, img2col_py_overlap,\
    col2im_CS_py, col2im_CS_py_overlap, psnr
import math
from torch.utils.data import Dataset, DataLoader
import csdata_fast
from torch.nn import init
import torch.nn.functional as F

parser = ArgumentParser(description='Nest-DGIL+')
parser.add_argument('--model_name', type=str, default='Nest-DGIL-learnA', help='model name')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {10, 25, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='1', help='gpu index')
parser.add_argument('--data_dir', type=str, default='cs_train400_png', help='training data directory')
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--patch_size', type=int, default=33, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir_org', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set,Set11/Set68/Urban100HR')
parser.add_argument('--test_cycle', type=int, default=20, help='epoch number of each test cycle')
parser.add_argument('--overlap_size', type=str, default=6, help='overlap pixel: 0/6')
parser.add_argument('--mode', type=str, default='test', help='train or test')
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list


test_name = args.test_name
test_cycle = args.test_cycle

test_dir = os.path.join(args.data_dir_org, test_name)
filepaths = glob.glob(test_dir + '/*.*')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 32

class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

class BasicBlock(torch.nn.Module):
    def __init__(self, features=32):
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

    # def forward(self, x,  PhiWeight, PhiTWeight, PhiTb,lambda_step,x_step):
    def forward(self, yold, z, PhiWeight, PhiTWeight, PhiTb, lambda_step, soft_thr):
        x = yold - self.Sp(lambda_step) * PhiTPhi_fun(z, PhiWeight, PhiTWeight)\
            + self.Sp(lambda_step) * PhiTb
        x_input = x.view(-1, 1, 33, 33)

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
        return [x_pred]


# Define GIL
class GIL(torch.nn.Module):
    def __init__(self, LayerNo,n_input):
        super(GIL, self).__init__()
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))

        onelayer = []
        self.LayerNo = LayerNo
        self.Sp = nn.Softplus()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

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

    def forward(self, x,n_input):
        batchx = x
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batchx, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)

        xold = PhiTb
        yold = xold
        for i in range(self.LayerNo):
            # x = self.fcs[i](x,  PhiWeight, PhiTWeight, PhiTb,lambda_step[i],x_step[i])
            theta1_ = self.w_theta1 * i + self.b_theta1  # parameter calculate
            mu1_ = self.w_mu1 * i + self.b_mu1
            gamma_ = self.Sp(self.w_gamma * i + self.b_gamma)
            x_coef = torch.tensor([1 - gamma_, gamma_])
            gamma_s = F.softmax(x_coef)  # ensure coef is positive
            z = gamma_s[0] * xold + gamma_s[1] * yold

            [ynew] = self.fcs[i](yold, z, PhiWeight, PhiTWeight, PhiTb, mu1_, theta1_)

            xnew = gamma_s[0] * xold + gamma_s[1] * ynew
            xold = xnew
            yold = ynew

        x_final = xnew
        return [x_final, Phi]

model = GIL(layer_num,n_input)
model = nn.DataParallel(model)
model = model.to(device)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len
########################################################################
def get_val_result(model,mode,save_image_str, test_name, data_dir_org, overlap_size):
    model.eval()
    with torch.no_grad():
        test_set_path = os.path.join(data_dir_org, test_name)
        test_set_path = glob.glob(test_set_path + '/*.*')
        ImgNum = len(test_set_path)
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

        if not os.path.exists(save_image_str):
            os.makedirs(save_image_str)

        time_all = 0
        for img_no in range(ImgNum):
            # imgName = filepaths[img_no]
            imgName = test_set_path[img_no]
            Img = cv2.imread(imgName, 1)
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()
            Iorg_y = Img_yuv[:, :, 0]

            if overlap_size == 0:
                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
                Icol = img2col_py(Ipad, 33).transpose() / 255.0
            else:
                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py_overlap(Iorg_y, overlap_size)
                Icol = img2col_py_overlap(Ipad, 33, overlap_size).transpose() / 255.0

            Img_output = Icol
            start = time()
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor).view(-1, 1, 33, 33)
            batch_x = batch_x.to(device)
            [x_output,  Phi] = model(batch_x, n_input)

            end = time()
            time_all = time_all + (end - start)

            Prediction_value = x_output.view(-1, 1089).cpu().data.numpy()
            if overlap_size == 0:
                X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)
            else:
                X_rec = np.clip(
                    col2im_CS_py_overlap(Prediction_value.transpose(), row, col, row_new, col_new, overlap_size), 0, 1)

            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)
            # save image
            if mode == 'test':
                print("[%02d/%02d] Run time for %s is %.5f, PSNR is %.3f, SSIM is %.5f" % (
                    img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))
                Img_rec_yuv[:, :, 0] = X_rec * 255

                im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                name_str = imgName.split("/")
                name_str2 = name_str[2].split(".")
                cv2.imwrite("./%s/%s_PSNR_%.3f_SSIM_%.5f.png" % (
                    save_image_str, name_str2[0], rec_PSNR, rec_SSIM), im_rec_rgb)

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM

        mean_time = time_all / ImgNum
    return PSNR_All,SSIM_All, mean_time
##########################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/%s_layer_%d_ratio_%d_lr_%.4f" % (args.model_dir, args.model_name, layer_num, args.cs_ratio, learning_rate)

log_file_name = "./%s/%s_Log_layer_%d_ratio_%d_lr_%.4f.txt" % (args.log_dir, args.model_name, layer_num, args.cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if start_epoch > 0:
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))

Eye_I = torch.eye(n_input).to(device)

if args.mode == 'train':
    training_data = csdata_fast.SlowDataset(args)
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8,
                             shuffle=True)
    # Training loop
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        for data in rand_loader:

            batch_x = data.view(-1, 1, 33, 33)
            batch_x = batch_x.to(device)

            [x_output, Phi] = model(batch_x, n_input)
            # Compute and print loss
            loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))

            # loss_orth = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1)) - Eye_I, 2))

            loss_all = loss_discrepancy # + 0.01 * loss_constraint # +0.01*loss_constraint

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()


        output_loss = str(datetime.now()) + " [%d/%d] Total loss: %.4f, discrepancy loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
        print(output_loss)

        # write_data(log_file_name, output_data)
        PSNR_All, SSIM_All, mean_time = get_val_result(model, args.mode, model_dir,  test_name,
                                                       args.data_dir_org, args.overlap_size)
        output_data = "CS ratio is %d, Avg time is %.5f, Avg Proposed PSNR/SSIM is %.3f/%.5f, Epoch number of model is %d \n" % (
            args.cs_ratio, mean_time, np.mean(PSNR_All), np.mean(SSIM_All), epoch_i)
        print(output_data)

        # save result
        output_data = [epoch_i, np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All)]
        output_file = open(model_dir + "/log_PSNR.txt", 'a')
        for fp in output_data:  # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')  # line feed
        output_file.close()

        if epoch_i % test_cycle == 0:
            print('model saved!')
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

elif args.mode == 'test':
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, args.end_epoch)))
    model_dir_result = "./%s/CS_%s_ratio_%d" % (args.result_dir, args.model_name, args.cs_ratio)
    print('Test')
    PSNR_All, SSIM_All, mean_time = get_val_result(model, args.mode, model_dir_result,
                                                   test_name, args.data_dir_org, args.overlap_size)
    output_data = "CS ratio is %d, Avg time is %.5f, Avg Proposed PSNR/SSIM is %.3f/%.5f, Epoch number of model is %d \n" % (
        args.cs_ratio, mean_time, np.mean(PSNR_All), np.mean(SSIM_All), args.end_epoch)
    print(output_data)

