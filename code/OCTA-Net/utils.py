# -*- coding: utf-8 -*-

import os
import visdom
import numpy as np
import time


def mkdir(path):
    # 引入模块
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1.0 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 定义获取当前学习率的函数
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 构建数据集，可扩展
def build_dataset(dataset, data_dir, channel=1, isTraining=True, crop_size=(64, 64), scale_size=(512, 512)):
    if dataset == "rose":
        from octa_dataset import ROSE
        database = ROSE(data_dir, channel=channel, isTraining=isTraining)
    elif dataset == "cria":
        from octa_dataset import CRIA
        database = CRIA(data_dir, channel=channel, isTraining=isTraining, scale_size=scale_size)
    elif dataset == "drive":
        from fundus_dataset import DRIVE
        database = DRIVE(data_dir, channel=channel, isTraining=isTraining, scale_size=scale_size)
    else:
        raise NotImplementedError('dataset [%s] is not implemented' % dataset)
    
    return database


# 构建模型，可扩展
def build_model(model, device, channel=1):
    if model == "unet":
        from other_models import U_Net
        net = U_Net(img_ch=channel, output_ch=1).to(device)
    elif model == "cenet":
        print("input channel of CE-Net must be 3, param channel no used")
        from imed_models import CE_Net
        net = CE_Net(num_classes=1).to(device)
    elif model == "resunet":
        from other_models import ResUNet
        net = ResUNet(img_ch=channel, output_ch=1).to(device)
    elif model == "csnet":
        from imed_models import CS_Net
        net = CS_Net(in_channels=channel, out_channels=1).to(device)
    elif model == "srfunet":
        from other_models import SRF_UNet
        net = SRF_UNet(img_ch=channel, output_ch=1).to(device)
    else:
        raise NotImplementedError('model [%s] is not implemented' % model)
    
    return net


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """
    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        # 画的第几个数，相当于横坐标
        # 比如("loss", 23) 即loss的第23个点
        self.index = {}
        self.log_text = ""
    
    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        
        return self
    
    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ("loss", 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)
    
    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)
    
    def plot(self, name, y, **kwargs):
        # self.plot("loss", 1.00)
        
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else "append",
                      **kwargs
                      )
        self.index[name] = x + 1
    
    def img(self, name, img_, **kwargs):
        """
        self.img("input_img", t.Tensor(64, 64))
        self.img("input_imgs", t.Tensor(3, 64, 64))
        self.img("input_imgs", t.Tensor(100, 1, 64, 64))
        self.img("input_imgs", t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )
    
    def log(self, info, win="log_text"):
        """
        self.log({"loss": 1, "lr": 0.0001})
        """
        self.log_text += ("[{time}] {info} <br>".format(
            time=time.strftime("%m%d_%H%M%S"), info=info))
        self.vis.text(self.log_text, win)
    
    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot, image, log, plot_many等除外
        """
        return getattr(self.vis, name)
