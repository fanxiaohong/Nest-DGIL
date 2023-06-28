import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,TensorDataset
import scipy.io as sio
import h5py
####################################################################################
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
# ####################################################################################
# ct train dataloder
def CT_dataloader(num_work,batch_size,run_mode):
    # train data
    if run_mode == 'train':
        train_data_Name = './data/CT/HU/train/full_sampled1.mat'
        train_full_sampled_data = sio.loadmat(train_data_Name)
        train_full_sampled_matrix = train_full_sampled_data['image_all']
        for i in range(7):  # eight files train labels
            train_full_sampled_data_Name = './data/CT/HU/train/full_sampled' + str(i + 2) + '.mat'
            train_full_sampled_data = sio.loadmat(train_full_sampled_data_Name)
            train_full_sampled_matrix_tmp = train_full_sampled_data['image_all']
            train_full_sampled_matrix = np.concatenate((train_full_sampled_matrix, train_full_sampled_matrix_tmp))
        print('Train full-sampled data shape', np.array(train_full_sampled_matrix).shape)
        train_loader = DataLoader(dataset=RandomDataset(train_full_sampled_matrix, train_full_sampled_matrix.shape[0]), batch_size=batch_size, num_workers=num_work,
                                      shuffle=True)

        # val data
        val_data_Name = './data/CT/HU/val/full_sampled.mat'
        val_full_sampled_data = sio.loadmat(val_data_Name)
        val_full_sampled_matrix = val_full_sampled_data['image_all']
        print('Val full_sampled_data shape', np.array(val_full_sampled_matrix).shape)
        val_loader = DataLoader(dataset=RandomDataset(val_full_sampled_matrix, val_full_sampled_matrix.shape[0]),
                                  batch_size=1, num_workers=num_work,shuffle=True)

    # test data
    test_data_Name = './data/CT/HU/test/full_sampled.mat'
    test_full_sampled_data = sio.loadmat(test_data_Name)
    test_full_sampled_matrix = test_full_sampled_data['image_all']
    print('Test full_sampled data shape', np.array(test_full_sampled_matrix).shape)
    test_loader = DataLoader(dataset=RandomDataset(test_full_sampled_matrix, test_full_sampled_matrix.shape[0]),
                            batch_size=1, num_workers=num_work)
    if run_mode == 'train':
        return train_loader,val_loader,test_loader
    elif run_mode == 'test':
        return test_loader,test_loader, test_loader
