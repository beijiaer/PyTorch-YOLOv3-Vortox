"""
===============================
  -*- coding:utf-8 -*-
  Author    :hanjinyue
  Time      :2020/12/16 20:24
  File      :hello.py
================================
"""
import tensorflow
print(tensorflow.__file__)
import torch

print("Hello World, Hello PyTorch {}".format(torch.__version__))

print("\nCUDA is available:{}, version is {}".format(torch.cuda.is_available(), torch.version.cuda))

print("\ndevice_name: {}".format(torch.cuda.get_device_name(0)))

str = '../PytorchYOLOv3/checkpoints/yolov3_vortox_18.pth'
str_list = int(str.split('_')[-1][0:-4])+1

print(str_list)