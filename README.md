# Nest-DGIL: Nesterov-optimized Deep Geometric Incremental Learning for CS Image Reconstruction

This repository contains the natural image CS and sparse-view CT reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Ke Chen, Yujie Feng, and Jianping Zhang*, “Nest-DGIL: Nesterov-optimized Deep Geometric Incremental Learning for CS Image Reconstruction”, IEEE Transactions on Computational Imaging, vol. 9, pp. 819-833, Sep. 2023, doi: 10.1109/TCI.2023.3315853.[[pdf]](https://ieeexplore.ieee.org/document/10252020) 

Xiaohong Fan, Yin Yang, Ke Chen, Yujie Feng, and Jianping Zhang*, “Nest-DGIL: Nesterov-optimized Deep Geometric Incremental Learning for CS Image Reconstruction”, arXiv, August 2023. [[pdf]](https://arxiv.org/pdf/2308.03807) 

### Abstract
Proximal gradient-based optimization is one of the most common strategies to solve inverse problem of images, and it is easy to implement. However, these techniques often generate heavy artifacts in image reconstruction. One of the most popular refinement methods is to fine-tune the regularization parameter to alleviate such artifacts, but it may not always be sufficient or applicable due to increased computational costs. In this work, we propose a deep geometric incremental learning framework based on the second Nesterov proximal gradient optimization. The proposed end-to-end network not only has the powerful learning ability for high-/low-frequency image features, but also can theoretically guarantee that geometric texture details will be reconstructed from preliminary linear reconstruction. Furthermore, it can avoid the risk of intermediate reconstruction results falling outside the geometric decomposition domains and achieve fast convergence. Our reconstruction framework is decomposed into four modules including general linear reconstruction, cascade geometric incremental restoration, Nesterov acceleration, and post-processing. In the image restoration step, a cascade geometric incremental learning module is designed to compensate for missing texture information from different geometric spectral decomposition domains. Inspired by the overlap-tile strategy, we also develop a post-processing module to remove the block effect in patch-wise-based natural image reconstruction. All parameters in the proposed model are learnable, an adaptive initialization technique of physical parameters is also employed to make model flexibility and ensure converging smoothly. We compare the reconstruction performance of the proposed method with existing state-of-the-art methods to demonstrate its superiority. Our source codes are available at https://github.com/fanxiaohong/Nest-DGIL .

![Fig1_NestDGIL_Chart_sc_00](https://github.com/fanxiaohong/Nest-DGIL/assets/48355877/ed6d8e80-38d3-4ef4-8c9e-c012fa02d68a)
Fig. 1. The overall architecture of our Nest-DGIL network. It consists of four main modules, i.e. linear reconstruction module Dk, cascade geometric
incremental learning module Pk, Nesterov acceleration module Nk and post-processing module.

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and Nvidia Tesla V100 GPU.

### Environment  
```
pytorch <= 1.7.1 (recommend 1.6.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.16.1, 0.16.2)
torch-radon = 1.0.0 (for sparse-view CT)
```

### 1.Test natural image CS    
1.1、Pre-trained models:  
All pre-trained models for our paper are in './model'.  
1.2、Prepare test data:  
The original test sets are in './data/'.  
1.3、Prepare code:  
Open './Core-Nest-DGIL-natural-W-CS25.py' and change the default run_mode to test in parser (parser.add_argument('--mode', type=str, default='test', help='train or test')).  
1.4、Run the test script (Core-Nest-DGIL-natural-W-CS25.py).  
1.5、Check the results in './result/'.

### 2.Train natural image CS  
2.1、Prepare training data:  
We use the same datasets and training data pairs as ISTA-Net++ for CS. Due to upload file size limitation, we are unable to upload training data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.  
2.2、Prepare measurement matrix:  
The measurement matrixs are in './sampling_matrix/'.  
2.3、Prepare code:  
Open '.Core-Nest-DGIL-natural-W-CS25.py' and change the default run_mode to train in parser (parser.add_argument('--mode', type=str, default='train', help='train or test')).  
2.4、Run the train script (Core-Nest-DGIL-natural-W-CS25.py).  
2.5、Check the results in './log/'.

### 3.Test sparse-view CT  
The torch-radon package (pip install torch-radon) is necessary for sparse-view CT reconstruction.    
3.1、Pre-trained models:  
All pre-trained models for our paper are in './model_CT'.  
3.2、Prepare test data:  
Due to upload file size limitation, we are unable to upload testing data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.   
3.3、Prepare code:  
Open './Core_Nest-DGIL-CT-ds12.py' and change the default run_mode to test in parser (parser.add_argument('--run_mode', type=str, default='test', help='train or test')).  
3.4、Run the test script (Core_Nest-DGIL-CT-ds12.py).  
3.5、Check the results in './result/'.

### 4.Train sparse-view CT   
4.1、Prepare training data:  
Due to upload file size limitation, we are unable to upload training data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.  
4.2、Prepare code:  
Open '.Core_Nest-DGIL-CT-ds12.py' and change the default run_mode to train in parser (parser.add_argument('--run_mode', type=str, default='train', help='train or test')).  
4.3、Run the train script (Core_Nest-DGIL-CT-ds12.py).  
4.4、Check the results in './log_CT/'.

### Citation  
If you find the code helpful in your resarch or work, please cite the following papers. 
```
@Article{Fan2023,
  author  = {Xiaohong Fan and Yin Yang and Ke Chen and Yujie Feng and Jianping Zhang},
  journal = {IEEE Transactions on Computational Imaging},
  title   = {Nest-DGIL: Nesterov-optimized Deep Geometric Incremental Learning for CS Image Reconstruction},
  year    = {2023},
  month   = {Sep.},
  pages   = {819--833},
  volume  = {9},
  doi     = {10.1109/TCI.2023.3315853},
}
```

### Acknowledgements  
Thanks to the authors of ISTA-Net++, CSformer and FISTA-Net, our codes are adapted from the open source codes of them.   

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong1992@gmail.com or fanxiaohong@smail.xtu.edu.cn
