# Meta-SR
Official implementation of **Meta-SR: A Magnification-Arbitrary Network for Super-Resolution(CVPR2019)(PyTorch)**
  
[Paper](https://arxiv.org/pdf/1903.00875.pdf)

Our code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).


# Attention
I find an error in my camera-ready, the PSNR of our Meta-RDN on scale 1.2 is 40.04 not 40.40.
# Requirements

* Pytorch 0.4.0
* Python 3.5
* numpy
* skimage
* imageio
* cv2  
*note that if you use another version of pytorch (>0.4.0), you can rewrite the dataloader.py

# Update notes
* 2019/12/23: fix a bug in https://github.com/XuecaiHu/Meta-SR-Pytorch/blob/f2cf094248defef242973282627ac8ea50d2e806/trainer.py#L107 , since the zeros in the double data type isnot a real zero.

* 2019/12/06:   I rewrite the input_matrix_wpn function in trainer.py. Since the offset is repeated, there are many repeated weight prediction. I remove them. In the metardn, we use repeated_weights to extend a small matrix to a matrix with same size of the feature maps.  This version use less memory and less inference times.
* todo provide code for pytorch 1.0, 1.1 and 1.2


# Install and run demo
1. download the code
```
git clone https://github.com/XuecaiHu/Meta-SR-Pytorch.git
cd Meta-SR-Pytorch
```


2. run training demo:
```
python main.py --model metardn --ext sep  --save metardn --lr_decay 200 --epochs 1000 --n_GPUs 1 --batch_size 1
```

3. run test demo:
* download the model from the [BaiduYun](https://pan.baidu.com/s/14L4Aut-F4JoSRfkJh6vr4Q) fetch code: btc5. 
* put the model_1000.pt under the ./eperiment/metardn/model/

```
python main.py --model metardn --ext sep  --save metardn --n_GPUs 1 --batch_size 1 --test_only --data_test Set5 --pre_train  ./experiment/metardn/model/model_1000.pt  --save_results --scale 1.5
```

# Train and Test as our paper

1.  prepare  dataset
   * download the dataset [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) and [test dataset](https://pan.baidu.com/s/1tzJFzEN5QdN53CcE1DheHw) fetch code: ev7u [GoogleDrive](https://drive.google.com/open?id=14BW1q3_i6FRoq7PwwQ-81GbXWph6934x)
   *  change the path_src = DIV2K HR image folder path and run /prepare_dataset/geberate_LR_metasr_X1_X4.m 
   *  upload the dataset 
   * change the  dir_data in option.py： dir_data = "/path to your DIV2K and testing dataset'(keep the training and test dataset in the same folder: test dataset under the benchmark folder and training dataset rename to DIV2K, or change the data_train to your folder name)  
2. pre_train model  for test
  [BaiduYun](https://pan.baidu.com/s/14L4Aut-F4JoSRfkJh6vr4Q) fetch code: btc5  
  [GoogleDrive](https://drive.google.com/open?id=1tGjz_pzgvo1T2N4f_ZjuqmxQHdpeDiSB)
  
## train 
```
cd /Meta-SR-Pytorch 
python main.py --model metardn --save metardn --ext sep --lr_decay 200 --epochs 1000 --n_GPUs 4 --batch_size 16 
```
## test 
```
python main.py --model metardn --save metardn --ext sep --pre_train ./experiment/metardn/model/model_1000.pt --test_only --data_test Set5  --scale 1.5 --n_GPUs 1
```
# Citation
```
@article{hu2019meta,
  title={Meta-SR: A Magnification-Arbitrary Network for Super-Resolution},
  author={Hu, Xuecai and Mu, Haoyuan and Zhang, Xiangyu and Wang, Zilei  and Tan, Tieniu and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
# Contact
Xuecai Hu (huxc@mail.ustc.edu.cn)
