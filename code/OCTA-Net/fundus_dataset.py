# -*- coding: utf-8 -*-

"""
读取图像统一用PIL而非cv2
"""
import os
import cv2
import time  # ##
import random
from PIL import Image
import numpy as np

import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF


# 灰度线性拉伸
def linear_stretch(img_arr):
    return 255.0 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))


# 裁剪，保证image, label和mask的裁剪方式一致
def Crop(image, label, mask, max_attempts=3, crop_size=(64, 64)):
    # Crop
    count = 0
    while True:
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        if (np.array(TF.crop(mask, i, j, h, w)) == 255).all():
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            break
        else:
            count += 1
            if count == max_attempts:
                crop_obj = transforms.CenterCrop(crop_size)
                image = crop_obj(image)
                label = crop_obj(label)
                mask = crop_obj(mask)
                break
    
    return image, label, mask


# Extend the full image because patch divison is not exact
def paint_border_overlap(full_img, crop_size=(64, 64), stride=(5, 5)):
    assert len(full_img.shape) == 2 or len(full_img.shape) == 3
    if len(full_img.shape) == 3:
        assert full_img.shape[2] == 3
    assert len(crop_size) == 2 and len(stride) == 2
    
    img_h = full_img.shape[0]  # height of the full image
    img_w = full_img.shape[1]  # width of the full image
    leftover_h = (img_h - crop_size[0]) % stride[0]  # leftover on the h dim
    leftover_w = (img_w - crop_size[1]) % stride[1]  # leftover on the w dim
    
    if (leftover_h != 0):  # change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " + str(stride[0]))
        print("img_h " + str(img_h) + ", patch_h " + str(crop_size[0]) + ", stride_h " + str(stride[0]))
        print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        print("So the H dim will be padded with additional " + str(stride[0] - leftover_h) + " pixels")
        
        if len(full_img.shape) == 3:
            full_img = np.pad(full_img, ((0, stride[0] - leftover_h), (0, 0), (0, 0)))
        else:
            full_img = np.pad(full_img, ((0, stride[0] - leftover_h), (0, 0)))
    
    if (leftover_w != 0):   # change dimension of img_w
        print("the side W is not compatible with the selected stride of " + str(stride[1]))
        print("img_w " + str(img_w) + ", patch_w " + str(crop_size[1]) + ", stride_w " + str(stride[1]))
        print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        print("So the W dim will be padded with additional " + str(stride[1] - leftover_w) + " pixels")
        
        if len(full_img.shape) == 3:
            full_img = np.pad(full_img, ((0, 0), (0, stride[1] - leftover_w), (0, 0)))
        else:
            full_img = np.pad(full_img, ((0, 0), (0, stride[1] - leftover_w)))
    
    print("new full image shape: \n" + str(full_img.shape))
    
    return full_img


# Divide all the full_img in pacthes
def extract_ordered_overlap(full_img, crop_size=(64, 64), stride=(5, 5)):
    assert len(full_img.shape) == 2 or len(full_img.shape) == 3
    if len(full_img.shape) == 3:
        assert full_img.shape[2] == 3
    assert len(crop_size) == 2 and len(stride) == 2
    
    img_h = full_img.shape[0]  # height of the full image
    img_w = full_img.shape[1]  # width of the full image
    assert (img_h - crop_size[0]) % stride[0] == 0 and (img_w - crop_size[1]) % stride[1] == 0
    N_patches_img = ((img_h - crop_size[0]) // stride[0] + 1) * ((img_w - crop_size[1]) // stride[1] + 1)  # // --> division between integers
    # N_patches_tot = N_patches_img * full_img.shape[0]
    
    print("Number of patches on h : " + str(((img_h - crop_size[0]) // stride[0] + 1)))
    print("Number of patches on w : " + str(((img_w - crop_size[1]) // stride[1] + 1)))
    print("number of patches in the image: " + str(N_patches_img))  # + ", totally for this dataset: " + str(N_patches_tot))
    # patches = np.empty((N_patches_tot, full_img.shape[1], patch_h, patch_w))
    
    # [N_patches_img, channel, patch_h, patch_w]
    if len(full_img.shape) == 3:
        patches = np.empty((N_patches_img, 3, crop_size[0], crop_size[1]), np.uint8)
    else:
        patches = np.empty((N_patches_img, 1, crop_size[0], crop_size[1]), np.uint8)
    
    iter_tot = 0   # iter over the total number of patches (N_patches)
    # for i in range(full_img.shape[0]):  # loop over the full image
    for h in range((img_h - crop_size[0]) // stride[0] + 1):
        for w in range((img_w - crop_size[1]) // stride[1] + 1):
            if len(full_img.shape) == 3:
                patches[iter_tot, 0, :, :] = full_img[h*stride[0]:h*stride[0]+crop_size[0], w*stride[1]:w*stride[1]+crop_size[1], 0]
                patches[iter_tot, 1, :, :] = full_img[h*stride[0]:h*stride[0]+crop_size[0], w*stride[1]:w*stride[1]+crop_size[1], 1]
                patches[iter_tot, 2, :, :] = full_img[h*stride[0]:h*stride[0]+crop_size[0], w*stride[1]:w*stride[1]+crop_size[1], 2]
            else:
                patches[iter_tot, 0, :, :] = full_img[h*stride[0]:h*stride[0]+crop_size[0], w*stride[1]:w*stride[1]+crop_size[1]]
            iter_tot += 1   # total
    assert iter_tot == N_patches_img  # N_patches_tot
    
    return patches  # array with all the full_img divided in patches


# Load the original data and return the extracted patches for testing
def get_data_testing_overlap(img, crop_size=(64, 64), stride=(5, 5)):
    assert len(crop_size) == 2 and len(stride) == 2
    start_t = time.time()  # ##
    img = np.array(img, dtype=np.uint8)
    full_img = paint_border_overlap(img, crop_size, stride)
    
    print("\ntest image shape:")
    print(full_img.shape)
    print("test image range (min - max): " + str(np.min(full_img)) + ' - ' + str(np.max(full_img)))
    print("\npaint_border_overlap: %f s\n" % (time.time() - start_t))  # ##
    start_t = time.time()  # ##
    # extract the TEST patches from the full image
    patches_img_test = extract_ordered_overlap(full_img, crop_size, stride)
    
    print("\ntest PATCHES image shape:")
    print(patches_img_test.shape)
    print("test PATCHES image range (min - max): " + str(np.min(patches_img_test)) + ' - ' + str(np.max(patches_img_test)))
    print("\nextract_ordered_overlap: %f s\n" % (time.time() - start_t))  # ##
    
    return patches_img_test, full_img.shape[0], full_img.shape[1]


def recompone_overlap(preds, full_shape, stride=(5, 5)):
    assert len(preds.shape) == 4  # 4D arrays
    assert preds.shape[1] == 1  # check the channel is 1
    assert len(full_shape) == 2 and len(stride) == 2
    start_t = time.time()  # ##
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (full_shape[0] - patch_h) // stride[0] + 1
    N_patches_w = (full_shape[1] - patch_w) // stride[1] + 1
    N_patches_img = N_patches_h * N_patches_w
    assert preds.shape[0] == N_patches_img
    
    print("N_patches_h: " + str(N_patches_h))
    print("N_patches_w: " + str(N_patches_w))
    print("N_patches_img: " + str(N_patches_img))
    
    # itialize to zero mega array with sum of Probabilities
    full_prob = np.zeros((1, preds.shape[1], full_shape[0], full_shape[1]))
    full_sum = np.zeros((1, preds.shape[1], full_shape[0], full_shape[1]))
    
    k = 0  # iterator over all the patches
    for h in range((full_shape[0] - patch_h) // stride[0] + 1):
        for w in range((full_shape[1] - patch_w) // stride[1] + 1):
            full_prob[0, :, h*stride[0]:h*stride[0]+patch_h, w*stride[1]:w*stride[1]+patch_w] += preds[k]
            full_sum[0, :, h*stride[0]:h*stride[0]+patch_h, w*stride[1]:w*stride[1]+patch_w] += 1
            k += 1
    assert k == preds.shape[0]
    assert np.min(full_sum) >= 1.0  # at least one
    final_avg = full_prob / full_sum
    print("\nrecompone_overlap: %f s\n" % (time.time() - start_t))  # ##
    assert np.max(final_avg) <= 1.0  # max value for a pixel is 1.0
    assert np.min(final_avg) >= 0.0  # min value for a pixel is 0.0
    
    return final_avg


class DRIVE(data.Dataset):
    def __init__(self, root, channel=3, isTraining=True, scale_size=(512, 512)):
        super(DRIVE, self).__init__()
        self.img_lst, self.gt_dct, self.mask_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
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
        self.name = imgPath.split("/")[-1][0:2] + ".tif"
        gtPath = self.gt_dct["gt"][index]
        maskPath = self.mask_lst[index]
        
        simple_transform = transforms.ToTensor()
        
        img = Image.open(imgPath)
        w, h = img.size
        img = img.resize(self.scale_size, Image.BICUBIC)
        gt = Image.open(gtPath).convert("L")
        mask = Image.open(maskPath).convert("L")
        gt = gt.resize(self.scale_size, Image.BICUBIC)
        mask = mask.resize(self.scale_size, Image.BICUBIC)
        
        gt = np.array(gt, dtype=np.uint8)
        gt[gt >= 128] = 255
        gt[gt < 128] = 0
        gt = Image.fromarray(gt)
        
        mask = np.array(mask, dtype=np.uint8)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        mask = Image.fromarray(mask)
        
        if self.channel == 1:
            img = img.convert("L")
            img = np.array(img, dtype=np.uint8)
            img = linear_stretch(img)  # 灰度线性拉伸
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(np.array(img, dtype=np.uint8))  # CLAHE
            img = np.array(img, dtype=np.uint8)
            img = Image.fromarray(img)
            img = TF.adjust_gamma(img, gamma=1.2, gain=1)  # gamma校正
            img_transform = transforms.ToTensor()
        else:
            img = img.convert("RGB")
            img = np.array(img, dtype=np.uint8)
            img = Image.fromarray(img)
            img_transform = transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),
                # transforms.ColorJitter(brightness=0.7, contrast=0.6, saturation=0.5),  # 随机改变图片的亮度、对比度和饱和度
                transforms.ToTensor(),
            ])
        
        if "manual" in self.gt_dct:  # test
            manualPath = self.gt_dct["manual"][index]
            manual = Image.open(manualPath).convert("L").resize(self.scale_size, Image.BICUBIC)



            manual = np.array(manual, dtype=np.uint8)
            manual[manual >= 128] = 255
            manual[manual < 128] = 0
            manual = Image.fromarray(manual)
            
            img = simple_transform(img)
            gt = simple_transform(gt)
            mask = simple_transform(mask)
            manual = simple_transform(manual)
            
            return img, gt, mask, manual, (w, h)
        else:  # training
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
            mask = mask.rotate(angel)
            # img, gt, mask = Crop(img, gt, mask, max_attempts=3, crop_size=self.crop_size)

            img = img_transform(img)
            gt = simple_transform(gt)
            mask = simple_transform(mask)
            
            return img, gt, mask
    
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
        gt_dct = {}
        if isTraining:
            img_dir = os.path.join(root + "/training/images")
            gt_dir = os.path.join(root + "/training/label")
            mask_dir = os.path.join(root + "/training/mask")
        else:
            img_dir = os.path.join(root + "/test/images")
            gt_dir = os.path.join(root + "/test/label")
            mask_dir = os.path.join(root + "/test/mask")
            manual_dir = os.path.join(root + "/test/2nd_manual")
            manual_lst = sorted(list(map(lambda x: os.path.join(manual_dir, x), os.listdir(manual_dir))))
            gt_dct["manual"] = manual_lst
        
        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        mask_lst = sorted(list(map(lambda x: os.path.join(mask_dir, x), os.listdir(mask_dir))))
        
        gt_dct["gt"] = gt_lst
        assert len(img_lst) == len(mask_lst) == len(gt_lst)
        if "manual" in gt_dct:
            assert len(gt_dct["manual"]) == len(gt_dct["gt"])
        
        return img_lst, gt_dct, mask_lst
    
    def getFileName(self):
        return self.name
