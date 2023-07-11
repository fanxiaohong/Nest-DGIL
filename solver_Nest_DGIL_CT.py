import os
import torch
import torchvision
import torch.nn as nn
import glob
from skimage.measure import compare_ssim as ssim
from functions import *
import cv2
from time import time
import matplotlib.pyplot as plt
import random
from torch_radon import Radon
import SimpleITK as sitk
###########################################################################
def get_cond(cs_ratio, sigma, type):
    para_cs = None
    para_noise = sigma / 5.0
    if type == 'org':
        para_cs = cs_ratio * 2.0 / 100.0
    elif type == 'org_ratio':
        para_cs = cs_ratio / 100.0

    para_cs_np = np.array([para_cs])
    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)
    para_cs = para_cs.cuda()
    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)
    para_noise = para_noise.cuda()
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)
    return para
###########################################################################
class Solver_CT(object):
    def __init__(self, model, train_loader,val_loader,test_loader, args, device,penalty_mode):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = args.learning_rate
        self.start_epoch = args.start_epoch
        self.end_epoch = args.end_epoch
        self.device = device
        self.loss_mode = args.loss_mode
        self.run_mode = args.run_mode
        self.CT_test_name = args.CT_test_name
        self.model_dir = args.model_dir
        self.net_name = args.net_name
        self.layer_num = args.layer_num
        self.log_dir = args.log_dir
        self.result_dir = args.result_dir
        self.ds_factor = args.ds_factor
        self.save_image_mode = args.save_image_mode
        self.end_epoch = args.end_epoch

        self.theta_label = np.linspace(0, np.pi, 720, endpoint=False)
        if args.together_mode == "all":
            self.train_ds_set = [24, 12, 8, 6, 4]
        elif args.together_mode == "single":
            self.train_ds_set = [args.ds_factor]

        if penalty_mode == 'Fista-net':
            self.optimizer = torch.optim.Adam([
                {'params': self.model.module.fcs.parameters()},
                {'params': self.model.module.w_theta, 'lr': 0.0001},
                {'params': self.model.module.b_theta, 'lr': 0.0001},
                {'params': self.model.module.w_mu, 'lr': 0.0001},
                {'params': self.model.module.b_mu, 'lr': 0.0001},
                {'params': self.model.module.w_rho, 'lr': 0.0001},
                {'params': self.model.module.b_rho, 'lr': 0.0001}],
                lr=self.lr, weight_decay=0.0001)
        elif penalty_mode == 'Weight':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=0.0001)
        elif penalty_mode == 'None':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        elif penalty_mode == 'Adam+amsgrad':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, amsgrad=True)
        elif penalty_mode == 'Adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        elif penalty_mode == 'Adamw+amsgrad':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, amsgrad=True)

    def train(self):
        # define save dir
        model_dir = "./%s/%s_layer_%d_lr_%f" % (
            self.model_dir, self.net_name, self.layer_num, self.lr)
        # Load pre-trained model with epoch number
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        output_file = "./%s" % (self.log_dir)
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        if self.start_epoch > 0:  # train stop and restart
            self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, self.start_epoch)))
            self.model.to(self.device)

        # Training loop
        best_psnr_ds24 = 0
        best_psnr_ds12 = 0
        best_psnr_ds8 = 0
        best_psnr_ds6 = 0
        best_psnr_ds4 = 0
        for epoch_i in range(self.start_epoch + 1, self.end_epoch + 1):
            self.model.train(True)
            # model = self.model.train()
            step = 0
            for data in self.train_loader:
                step = step + 1
                full_sampled_x = data
                full_sampled_x = torch.unsqueeze(full_sampled_x, 1).cpu().data.numpy()

                # data augment
                if random.random() < 0.5: # random_horizontal_flip
                    full_sampled_x = full_sampled_x[:,:,:, ::-1].copy()

                if random.random() < 0.5: # random_vertical_flip
                    full_sampled_x = full_sampled_x[:,:,::-1, :].copy()

                # test torch radon
                full_sampled_x = torch.from_numpy(full_sampled_x).to(self.device)

                ds_theta = np.random.choice(self.train_ds_set)
                theta = np.linspace(0, np.pi, int(720 / ds_theta), endpoint=False)
                cond = get_cond(ds_theta, 0.0, 'org_ratio')

                radon = Radon(full_sampled_x.shape[2], theta, det_count=729)
                sinogram = radon.forward(full_sampled_x)
                b_in = radon.filter_sinogram(sinogram)
                X_fbp = radon.backprojection(b_in)

                radon_label = Radon(full_sampled_x.shape[2], self.theta_label, det_count=729)
                sinogram_label = radon_label.forward(full_sampled_x)
                b_in_label = radon_label.filter_sinogram(sinogram_label)
                batch_x = torch.abs(radon_label.backprojection(b_in_label))

                [x_output, x_mid, layers_residual_x] = self.model(cond,X_fbp, b_in,theta,self.theta_label)

                if self.loss_mode == 'L1':  # simple L1
                    loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))  # Compute and print loss
                    loss_all = loss_discrepancy   # Compute and print loss
                    loss_discrepancy_k = loss_discrepancy

                elif self.loss_mode == 'L2':  # simple L1
                    train_loss = nn.MSELoss()
                    loss_discrepancy = train_loss(x_output, batch_x)  # Compute and print loss
                    loss_all = loss_discrepancy   # Compute and print loss
                    loss_discrepancy_k = loss_discrepancy

                elif self.loss_mode == 'Fista-net':  # simple L1
                    # Compute loss, data consistency and regularizer constraints
                    train_loss = nn.MSELoss()
                    loss_discrepancy = train_loss(x_output, batch_x)  # + l1_loss(pred, y_target, 0.3)
                    loss_constraint = 0
                    for k, _ in enumerate(x_mid, 0):
                        loss_constraint += torch.mean(torch.pow(x_mid[k], 2))

                    encoder_constraint = 0
                    for k, _ in enumerate(layers_residual_x, 0):
                        encoder_constraint += torch.mean(torch.abs(layers_residual_x[k]))

                    loss_all = loss_discrepancy + 0.01 * loss_constraint + 0.001 * encoder_constraint
                    loss_discrepancy_k = loss_discrepancy

                elif self.loss_mode == 'Ista-net':  # simple L1
                    # Compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                    loss_constraint = torch.mean(torch.pow(x_mid[0], 2))
                    for k in range(self.layer_num - 1):
                        loss_constraint += torch.mean(torch.pow(x_mid[k + 1], 2))

                    gamma = torch.Tensor([0.01]).to(self.device)

                    # loss_all = loss_discrepancy
                    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
                    loss_discrepancy_k = loss_discrepancy

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss_all.backward()
                self.optimizer.step()

                # step %100==0
                if step % 100 == 0:
                    output_data = "[%02d/%02d] Step:%.0f | ds_factor:%.0f | Total Loss: %.6f |" \
                                  " Discrepancy Loss: %.6f| K-sapce Loss: %.6f" % \
                                  (epoch_i, self.end_epoch, step, ds_theta, loss_all.item(),
                                   loss_discrepancy.item(), loss_discrepancy_k.item())
                    print(output_data)

            # val
            model = self.model.eval()
            # Load pre-trained model with epoch number
            with torch.no_grad():
                RMSE_total = []
                PSNR_total = []
                SSIM_total = []
                PSNR_FBP_total = []
                image_num = 0
                time_all = 0
                for data in self.val_loader:
                    image_num = image_num + 1
                    full_sampled_x = data
                    # test torch radon
                    full_sampled_x = torch.unsqueeze(full_sampled_x, 1).to(self.device)

                    # radon_label = Radon(full_sampled_x.shape[2], self.theta_label)
                    sinogram_label = radon_label.forward(full_sampled_x)
                    b_in_label = radon_label.filter_sinogram(sinogram_label)
                    batch_x = torch.abs(radon_label.backprojection(b_in_label))

                    start = time()
                    # radon = Radon(full_sampled_x.shape[2], theta)
                    sinogram = radon.forward(full_sampled_x)
                    b_in = radon.filter_sinogram(sinogram)
                    X_fbp = radon.backprojection(b_in)

                    [x_output, x_mid, layers_residual_x] = model(cond,X_fbp, b_in,theta,self.theta_label)
                    end = time()
                    time_all = time_all + end - start
                    batch_x = batch_x.cpu().data.numpy().reshape(batch_x.shape[2], batch_x.shape[3])
                    x_output = x_output.cpu().data.numpy().reshape(batch_x.shape[0], batch_x.shape[1])
                    X_fbp = X_fbp.cpu().data.numpy().reshape(batch_x.shape[0], batch_x.shape[1])
                    # x_output_k = layers_residual_x.cpu().data.numpy().reshape(batch_x.shape[0], batch_x.shape[1])
                    rec_PSNR = psnr(x_output * 255.0, batch_x * 255.0)
                    rec_PSNR_FBP = psnr(X_fbp * 255.0, batch_x * 255.0)
                    rec_SSIM = ssim(x_output, batch_x, data_range=1)
                    rec_RMSE = compute_measure(batch_x, x_output, 1)
                    PSNR_total.append(rec_PSNR)
                    PSNR_FBP_total.append(rec_PSNR_FBP)
                    # PSNR_total_k.append(rec_PSNR_k)
                    SSIM_total.append(rec_SSIM)
                    RMSE_total.append(rec_RMSE)

                PSNR_mean = np.array(PSNR_total).mean()
                PSNR_FBP_mean = np.array(PSNR_FBP_total).mean()
                # PSNR_k_mean = np.array(PSNR_total_k).mean()
                SSIM_mean = np.array(SSIM_total).mean()
                RMSE_mean = np.array(RMSE_total).mean()
            print(
                "ds factor is %.0f, Mean run time for val is %.6f, PSNR FBP is %.3f,Proposed PSNR is %.3f, SSIM is %.5f, RMSE is %.5f" % (
                    ds_theta, time_all / image_num, PSNR_FBP_mean, PSNR_mean, SSIM_mean, RMSE_mean))

            # save model in every epoch
            if ds_theta == 24:
                if PSNR_mean > best_psnr_ds24:
                    print('PSNR_mean:{} > best_psnr:{}'.format(PSNR_mean, best_psnr_ds24))
                    best_psnr_ds24 = PSNR_mean
                    print('===========>save best model!')
                    torch.save(self.model.state_dict(),
                               "./%s/net_params_%d.pkl" % (model_dir, self.end_epoch))  # save only the parameters
            if ds_theta == 12:
                if PSNR_mean > best_psnr_ds12:
                    print('PSNR_mean:{} > best_psnr:{}'.format(PSNR_mean, best_psnr_ds12))
                    best_psnr_ds12 = PSNR_mean
                    print('===========>save best model!')
                    torch.save(self.model.state_dict(),
                               "./%s/net_params_%d.pkl" % (model_dir, self.end_epoch))
            if ds_theta == 8:
                if PSNR_mean > best_psnr_ds8:
                    print('PSNR_mean:{} > best_psnr:{}'.format(PSNR_mean, best_psnr_ds8))
                    best_psnr_ds8 = PSNR_mean
                    print('===========>save best model!')
                    torch.save(self.model.state_dict(),
                               "./%s/net_params_%d.pkl" % (model_dir, self.end_epoch))
            if ds_theta == 6:
                if PSNR_mean > best_psnr_ds6:
                    print('PSNR_mean:{} > best_psnr:{}'.format(PSNR_mean, best_psnr_ds6))
                    best_psnr_ds6 = PSNR_mean
                    print('===========>save best model!')
                    torch.save(self.model.state_dict(),
                               "./%s/net_params_%d.pkl" % (model_dir, self.end_epoch))
            if ds_theta == 4:
                if PSNR_mean > best_psnr_ds4:
                    print('PSNR_mean:{} > best_psnr:{}'.format(PSNR_mean, best_psnr_ds4))
                    best_psnr_ds4 = PSNR_mean
                    print('===========>save best model!')
                    torch.save(self.model.state_dict(),
                               "./%s/net_params_%d.pkl" % (model_dir, self.end_epoch))  # save only the parameters

            # save result
            output_data = [epoch_i, loss_all.item(),PSNR_FBP_mean, PSNR_mean, SSIM_mean, RMSE_mean]
            output_file_name = "./%s/%s_layer_%d_lr_%f.txt" % (
                self.log_dir, self.net_name, self.layer_num, self.lr)
            output_file = open(output_file_name, 'a')
            for fp in output_data:  # write data in txt
                output_file.write(str(fp))
                output_file.write(',')
            output_file.write('\n')  # line feed
            output_file.close()


    def test(self):
        # initial test file
        result_dir_tmp = os.path.join(self.result_dir, self.CT_test_name)
        result_dir = result_dir_tmp + '_' + self.net_name + '_ds_' + str(self.ds_factor) + '_epoch_' + str(
            self.end_epoch) + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Load pre-trained model with epoch number
        model_dir = "%s/%s_layer_%d_lr_%f" % (
            self.model_dir, self.net_name, self.layer_num, self.lr)
        self.model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, self.end_epoch)))

        model = self.model.eval()
        theta = np.linspace(0, np.pi, int(720 / self.ds_factor), endpoint=False)

        with torch.no_grad():
            RMSE_total = []
            PSNR_total = []
            # PSNR_total_k = []
            SSIM_total = []
            PSNR_FBP_total = []
            image_num = 0
            time_all = 0
            x_test_all = []
            for data in self.test_loader:
                image_num = image_num +1
                full_sampled_x = data
                # test torch radon
                full_sampled_x = torch.unsqueeze(full_sampled_x, 1).to(self.device)

                radon_label = Radon(full_sampled_x.shape[2], self.theta_label, det_count=729)
                sinogram_label = radon_label.forward(full_sampled_x)
                b_in_label = radon_label.filter_sinogram(sinogram_label)
                batch_x = torch.abs(radon_label.backprojection(b_in_label))

                start = time()
                radon = Radon(full_sampled_x.shape[2], theta, det_count=729)
                sinogram = radon.forward(full_sampled_x)
                b_in = radon.filter_sinogram(sinogram)
                X_fbp = radon.backprojection(b_in)

                cond = get_cond(self.ds_factor, 0.0, 'org_ratio')

                [x_output, x_mid, layers_residual_x] = model(cond,X_fbp, b_in,theta,self.theta_label)
                end = time()
                time_all = time_all + end -start
                batch_x = batch_x.cpu().data.numpy().reshape(batch_x.shape[2], batch_x.shape[3])
                x_output = x_output.cpu().data.numpy().reshape(batch_x.shape[0], batch_x.shape[1])
                X_fbp = X_fbp.cpu().data.numpy().reshape(batch_x.shape[0], batch_x.shape[1])

                # index calculate
                rec_PSNR = psnr(x_output * 255.0, batch_x * 255.0)
                rec_PSNR_FBP = psnr(X_fbp * 255.0, batch_x * 255.0)
                rec_SSIM = ssim(x_output, batch_x, data_range=1)
                rec_RMSE = compute_measure(batch_x, x_output, 1)
                PSNR_total.append(rec_PSNR)
                PSNR_FBP_total.append(rec_PSNR_FBP)
                SSIM_total.append(rec_SSIM)
                RMSE_total.append(rec_RMSE)

                # save image
                if self.save_image_mode == 1:
                    img_dir = result_dir + str(image_num) + "_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (
                        rec_PSNR, rec_SSIM, rec_RMSE)
                    print(img_dir)
                    im_rec_rgb = np.clip(x_output * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(img_dir, im_rec_rgb)
                    print("ds factor is %.0f, Mean run time for %s test is %.6f, Proposed PSNR is %.3f, SSIM is %.5f, Proposed RMSE is %.5f" % (
                        self.ds_factor, image_num, (end - start), rec_PSNR, rec_SSIM, rec_RMSE))

                elif self.save_image_mode == 2:
                    x_test_all.append(x_output)  # assemble

            if self.save_image_mode == 2:
                x_test_all = np.array(x_test_all)
                # 保存为nii, 可视化
                out = sitk.GetImageFromArray(x_test_all * 4096 - 1024)
                save_npy_name = result_dir + 'FBP_' + str(self.ds_factor) + '.nii.gz'
                sitk.WriteImage(out, save_npy_name)

            PSNR_FBP_mean = np.array(PSNR_FBP_total).mean()
            PSNR_FBP_std = np.array(PSNR_FBP_total).std()
            PSNR_mean = np.array(PSNR_total).mean()
            PSNR_std = np.array(PSNR_total).std()
            SSIM_mean = np.array(SSIM_total).mean()
            SSIM_std = np.array(SSIM_total).std()
            RMSE_mean = np.array(RMSE_total).mean()
            RMSE_std = np.array(RMSE_total).std()
            print("ds factor is %.0f, Mean run time for test is %.6f, Proposed FBP PSNR is %.3f-%.3f,"
                  " PSNR is %.3f-%.3f, "
                  "SSIM is %.5f-%.5f, RMSE is %.5f-%.5f" % (
                      self.ds_factor, time_all / image_num, PSNR_FBP_mean, PSNR_FBP_std,
                      PSNR_mean, PSNR_std, SSIM_mean, SSIM_std, RMSE_mean, RMSE_std))

        # save result
        output_file_name1 = result_dir + 'result_rmse_all.txt'
        output_data_rmse = [RMSE_total]
        output_file1 = open(output_file_name1, 'a')
        for fp in output_data_rmse:  # write data in txt
            output_file1.write(str(fp))
            output_file1.write(',')
        output_file1.write('\n')  # line feed
        output_file1.close()

        # save result
        output_data = [ self.end_epoch,PSNR_FBP_mean,PSNR_FBP_std, PSNR_mean,PSNR_std,
                        SSIM_mean,SSIM_std,RMSE_mean,RMSE_std]
        output_file_name = result_dir_tmp + '_' + self.net_name + '_ds_' + str(self.ds_factor) + '_epoch_' + str(
            self.end_epoch)+ ".txt"
        output_file = open(output_file_name, 'a')
        for fp in output_data:  # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')  # line feed
        output_file.close()


