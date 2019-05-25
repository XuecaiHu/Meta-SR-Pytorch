# Meta-SR
Official implementation of **Meta-SR: A Magnification-Arbitrary Network for Super-Resolution(CVPR2019)(PyTorch)

Our code is built on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).
# Requirements

* Pytorch 0.4.0
* Python 3.5
* numpy
* skimage
* imageio

# Train and Test
* prepare dataset: we follow the previous work, that is, we use the matlab imresize function to generate the LR images.
run the matlab files: 
* change the config: include dir_data

## train 
cd /Meta-SR-Pytorch  \n
python main.py --model metardn --save metardn_model_name --ext sep --lr_decay 200 --epochs 1000 

## test 
python main.py --model metardn --save metardn_model_name --ext sep --test_only --data_test BSD/Set14/Set5  --scale 2.3

# Citation
```
@article{hu2019meta,
  title={Meta-SR: A Magnification-Arbitrary Network for Super-Resolution},
  author={Hu, Xuecai and Mu, Haoyuan and Zhang, Xiangyu and Wang, Zilei  and Tan, Tieniu and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
