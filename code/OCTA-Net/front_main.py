# -*- coding: utf-8 -*-

import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import args
from utils import mkdir, build_dataset, Visualizer  # build_model,
from first_stage import SRF_UNet
# from second_stage import fusion
# from losses import build_loss
from train import train_first_stage
from val import val_first_stage
from test import test_first_stage


# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == "train":
    isTraining = True
else:
    isTraining = False

database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining,
                         crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
sub_dir = args.dataset + "/first_stage"  # + args.model + "/" + args.loss

if isTraining:  # train
    NAME = args.dataset + "_first_stage-2nd"  # + args.model + "_" + args.loss
    viz = Visualizer(env=NAME)
    writer = SummaryWriter(args.logs_dir + "/" + sub_dir)
    mkdir(args.models_dir + "/" + sub_dir)  # two stage时可以创建first_stage和second_stage这两个子文件夹
    
    # 加载数据集
    train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=False,
                                 crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
    val_dataloader = DataLoader(val_database, batch_size=1)
    
    # 构建模型
    first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1).to(device)
    first_net = torch.nn.DataParallel(first_net)
    first_optim = optim.Adam(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    
    # second_net = fusion(channels=args.base_channels, pn_size=args.pn_size, kernel_size=3, avg=0.0, std=0.1).to(device)
    # second_net = torch.nn.DataParallel(second_net)
    # second_optim = optim.Adam(second_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    
    thick_criterion = torch.nn.MSELoss()  # 可更改
    thin_criterion = torch.nn.MSELoss()  # 可更改
    fusion_criterion = torch.nn.MSELoss()  # 可更改
    
    best_thin = {"epoch": 0, "dice": 0}
    best_thick = {"epoch": 0, "dice": 0}
    best_fusion = {"epoch": 0, "dice": 0}
    # start training
    print("Start training...")
    for epoch in range(args.first_epochs):
        print('Epoch %d / %d' % (epoch + 1, args.first_epochs))
        print('-'*10)
        first_net = train_first_stage(viz, writer, train_dataloader, first_net, first_optim, args.init_lr, thin_criterion, thick_criterion, device, args.power, epoch, args.first_epochs)
        if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.first_epochs - 1:
            first_net, best_thin, best_thick, best_fusion = val_first_stage(best_thin, best_thick, best_fusion,
                                                            viz, writer, val_dataloader, first_net,
                                                            thin_criterion, thick_criterion, fusion_criterion, device,
                                                            args.save_epoch_freq, args.models_dir + "/" + sub_dir,
                                                            args.results_dir + "/" + sub_dir, epoch, args.first_epochs)
    print("Training finished.")
else:  # test
    # 加载数据集和模型
    test_dataloader = DataLoader(database, batch_size=1)
    net = torch.load(args.models_dir + "/" + sub_dir + "/front_model-" + args.first_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
    net.eval()
    
    # start testing
    print("Start testing...")
    test_first_stage(test_dataloader, net, device, args.results_dir + "/" + sub_dir, thin_criterion=None, thick_criterion=None, fusion_criterion=None,  isSave=True)
    print("Testing finished.")
