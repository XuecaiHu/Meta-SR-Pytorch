# Meta-SR
Official implementation of **Meta-SR: A Magnification-Arbitrary Network for Super-Resolution(CVPR2019)(PyTorch)**
  
[Paper](https://arxiv.org/pdf/1903.00875.pdf)

Our code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).
# Requirements

* Pytorch 0.4.0
* Python 3.5
* numpy
* skimage
* imageio
* cv2  
*note that if you use another version of pytorch (>0.4.0), you can rewrite the dataloader.py

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
   * download the dataset [DIV2K](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip) and [test dataset](https://pan.baidu.com/s/1-ccZyoKNBeo8yoiHMm3KHg) fetch code: hg69
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
