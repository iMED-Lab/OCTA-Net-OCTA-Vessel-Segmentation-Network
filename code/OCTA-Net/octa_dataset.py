# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
# import cv2
import random
from PIL import Image
import numpy as np

import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF


# 随机裁剪，保证image和label的裁剪方式一致
def random_crop(image, label, crop_size=(512, 512)):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = TF.crop(image, i, j, h, w)
    label = TF.crop(label, i, j, h, w)
    
    return image, label


class ROSE(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True):
        super(ROSE, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.name = ""
        
        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3
    
    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        # deepPath = self.deep_lst[index]
        # superficialPath = self.superficial_lst[index]
        
        simple_transform = transforms.ToTensor()
        
        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")
        # deep = Image.open(deepPath).convert("L")
        # superficial = Image.open(superficialPath).convert("L")
        
        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        
        gt = np.array(gt)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        gt = Image.fromarray(gt)
        
        # deep = np.array(deep)
        # deep[deep >= 128] = 255
        # deep[deep < 128] = 0
        # deep = Image.fromarray(deep)
        #
        # superficial = np.array(superficial)
        # superficial[superficial >= 128] = 255
        # superficial[superficial < 128] = 0
        # superficial = Image.fromarray(superficial)
        
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
            # deep = deep.rotate(angel)
            # superficial = superficial.rotate(angel)
        
        img = simple_transform(img)
        gt = simple_transform(gt)
        # deep = simple_transform(deep)
        # superficial = simple_transform(superficial)
        
        return img, gt
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)
    
    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/img")
            gt_dir = os.path.join(root + "/train/gt")
            deep_dir = os.path.join(root + "/train/thin_gt")
            superficial_dir = os.path.join(root + "/train/thick_gt")
        else:
            img_dir = os.path.join(root + "/test/img")
            gt_dir = os.path.join(root + "/test/gt")
            deep_dir = os.path.join(root + "/test/thin_gt")
            superficial_dir = os.path.join(root + "/test/thick_gt")
        
        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        # deep_lst = sorted(list(map(lambda x: os.path.join(deep_dir, x), os.listdir(deep_dir))))
        # superficial_lst = sorted(list(map(lambda x: os.path.join(superficial_dir, x), os.listdir(superficial_dir))))

        
        return img_lst, gt_lst
    
    def getFileName(self):
        return self.name


class CRIA(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(512, 512)):
        super(CRIA, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""
        
        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3
    
    def __getitem__(self, index):
        """
        内建函数，当对该类的实例进行类似字典的操作时，就会自动执行该函数，并返会对应的值
        这是必须要重载的函数，就是实现给定索引，返回对应的图像
        给出图像编号，返回变换后的输入图像和对应的label
        :param index: 图像编号
        :return:
        """
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()
        
        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")
        
        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        
        gt = np.array(gt)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        gt = Image.fromarray(gt)
        
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
        
        img = simple_transform(img)
        gt = simple_transform(gt)
        
        return img, gt
    
    def __len__(self):
        """
        返回总的图像数量
        :return:
        """
        return len(self.img_lst)
    
    def get_dataPath(self, root, isTraining):
        """
        依次读取输入图片和label的文件路径，并放到array中返回
        :param root: 存放的文件夹
        :return:
        """
        if isTraining:
            img_dir = os.path.join(root + "/train/original")
            gt_dir = os.path.join(root + "/train/gt")
        else:
            img_dir = os.path.join(root + "/test/original")
            gt_dir = os.path.join(root + "/test/gt")
        
        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        
        assert len(img_lst) == len(gt_lst)
        
        return img_lst, gt_lst
    
    def getFileName(self):
        return self.name
