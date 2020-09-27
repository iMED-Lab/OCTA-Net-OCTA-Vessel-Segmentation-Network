# -*- coding: utf-8 -*-

import os
import torch

from utils import mkdir, get_lr, adjust_lr
from test import test_first_stage


def train_first_stage(viz, writer, dataloader, net, optimizer, base_lr, thin_criterion, thick_criterion, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        thin_gt = sample[2].to(device)
        thick_gt = sample[3].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        thick_pred, thin_pred, _ = net(img)
        viz.img(name="images", img_=img[0, :, :, :])
        viz.img(name="thin labels", img_=thin_gt[0, :, :, :])
        viz.img(name="thick labels", img_=thick_gt[0, :, :, :])
        viz.img(name="thin prediction", img_=thin_pred[0, :, :, :])
        viz.img(name="thick prediction", img_=thick_pred[0, :, :, :])
        loss = thin_criterion(thin_pred, thin_gt) + thick_criterion(thick_pred, thick_gt)  # 可加权
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        viz.plot("train loss", loss.item())
        
        # 写入当前lr
        current_lr = get_lr(optimizer)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)
    
    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)
    
    return net


def train_second_stage(viz, writer, dataloader, front_net_thick, front_net_thin, fusion_net, optimizer, base_lr, criterion, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        with torch.no_grad(): 
            thick_pred = front_net_thick(img)
            thin_pred= front_net_thin(img)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        fusion_pred = fusion_net(img[:, :1, :, :], thick_pred, thin_pred)
        viz.img(name="images", img_=img[0, :, :, :])
        viz.img(name="labels", img_=gt[0, :, :, :])
        viz.img(name="prediction", img_=fusion_pred[0, :, :, :])
        loss = criterion(fusion_pred, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        viz.plot("train loss", loss.item())
        
        # 写入当前lr
        current_lr = get_lr(optimizer)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)
    
    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)
    
    return fusion_net
