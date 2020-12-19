# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @author     : yts3221@126.com
# @date       : 2019-08-21 10:08:00
# @brief      : 各数据集的Dataset定义
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
random.seed(1)
classes = ["person","background"]
#lesson34 fasterrcnn.py
class PennFudanDataset(Dataset):#han修改2020/12/3 21.42
#class PennFudanDataset(object):#Class RMBDataset(Dataset):
    def __init__(self, data_dir, transforms):

        self.data_dir = data_dir
        self.transforms = transforms
        self.img_dir = os.path.join(data_dir, "PNGImagesTemp")#实际图像文件夹
        self.txt_dir = os.path.join(data_dir, "Annotation")#图像注释文件夹
        #图像和注释具有相同的文件名，只是文件后缀不一样，names记录下所有文件名，不包括后缀
        self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".png"), os.listdir(self.img_dir)))]

    def __getitem__(self, index):
        """
        返回img和target
        :param idx:
        :return:
        """

        name = self.names[index]
        path_img = os.path.join(self.img_dir, name + ".png")
        path_xml = os.path.join(self.txt_dir, name + ".xml")#图像注释
        print("path_xml = ",path_xml)
        # load img：加载图像
        img = Image.open(path_img).convert("RGB")

        # load boxes and labels#加载图像注释描述
        file_xml = open(path_xml,'r')
        tree = ET.parse(file_xml)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        print("w = {},h = {}".format(w,h))
        boxes_list = list()
        labels_list = list()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            print("cls = {}".format(cls))
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            labels_list.append(cls_id)
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymax').text))
            boxes_list.append(box)
        boxes = torch.tensor(boxes_list, dtype=torch.float)
        labels = torch.tensor(labels_list, dtype=torch.long)
        target = {}
        target["boxes"] = boxes#(n,Xmin, Ymin,Xmax, Ymax)
        target["labels"] = labels#(n)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #返回结果：img是RGB图像
        #target是一个字典：target["boxes"] = boxes#(n,Xmin, Ymin,Xmax, Ymax)
        #target["labels"] = labels#(n)
        return img, target

    def __len__(self):
        if len(self.names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.names)


