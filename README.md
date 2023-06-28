# Nest-DGIL: Nesterov-optimized Deep Geometric Incremental Learning for Inverse Problems in CS Image Reconstruction

This repository contains the natural image CS and sparse-view CT reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Ke Chen, Yujie Feng, and Jianping Zhang*, “Nest-DGIL: Nesterov-optimized Deep Geometric Incremental Learning for Inverse Problems in CS Image Reconstruction”, 2023.  Under review.

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and Nvidia Tesla V100 GPU.

### Environment  
```
pytorch <= 1.7.1 (recommend 1.6.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.16.1, 0.16.2)
torch-radon = 1.0.0
```

### 1.Test natural image CS  
1.1、Pre-trained models:  
All pre-trained models for our paper are in './model'.  
1.2、Prepare test data:  
The original test sets are in './data/'.  
1.3、Prepare code:  
Open './Core-Nest-DGIL-natural-CS25.py' and change the default run_mode to test in parser (parser.add_argument('--mode', type=str, default='test', help='train or test')).  
1.4、Run the test script (Core-Nest-DGIL-natural-CS25.py).  
1.5、Check the results in './result/'.

### 2.Train natural image CS  
2.1、Prepare training data:  
We use the same datasets and training data pairs as ISTA-Net++ for CS. Due to upload file size limitation, we are unable to upload training data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.  
2.2、Prepare measurement matrix:  
The measurement matrixs are in './sampling_matrix/'.  
2.3、Prepare code:  
Open '.Core-Nest-DGIL-natural-CS25.py' and change the default run_mode to train in parser (parser.add_argument('--mode', type=str, default='train', help='train or test')).  
2.4、Run the train script (Core-Nest-DGIL-natural-CS25.py).  
2.5、Check the results in './log/'.

### 3.Test sparse-view CT  
1.1、Pre-trained models:  
All pre-trained models for our paper are in './model_CT'.  
1.2、Prepare test data:  
Due to upload file size limitation, we are unable to upload testing data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.   
1.3、Prepare code:  
Open './Core_Nest-DGIL-CT-ds12.py' and change the default run_mode to test in parser (parser.add_argument('--run_mode', type=str, default='test', help='train or test')).  
1.4、Run the test script (Core_Nest-DGIL-CT-ds12.py).  
1.5、Check the results in './result/'.

### 4.Train sparse-view CT   
2.1、Prepare training data:  
Due to upload file size limitation, we are unable to upload training data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.  
2.2、Prepare code:  
Open '.Core_Nest-DGIL-CT-ds12.py' and change the default run_mode to train in parser (parser.add_argument('--run_mode', type=str, default='train', help='train or test')).  
2.3、Run the train script (Core_Nest-DGIL-CT-ds12.py).  
2.4、Check the results in './log_CT/'.

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@smail.xtu.edu.cn
