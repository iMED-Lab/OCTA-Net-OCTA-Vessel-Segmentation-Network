# -*- coding: utf-8 -*-

import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import args
from utils import mkdir, build_dataset, Visualizer  # build_model,
from first_stage import SRF_UNet
from second_stage import fusion  # fusion
# from losses import build_loss
from train import train_second_stage
from val import val_second_stage
from test import test_second_stage


# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == "train":
    isTraining = True
else:
    isTraining = False

database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=isTraining,
                         crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
sub_dir = args.dataset + "/second_stage_v2"  # + args.model + "/" + args.loss

if isTraining:  # train
    NAME = args.dataset + "_second_stage_v2-2nd"  # + args.model + "_" + args.loss
    viz = Visualizer(env=NAME)
    writer = SummaryWriter(args.logs_dir + "/" + sub_dir)
    mkdir(args.models_dir + "/" + sub_dir)  # two stage时可以创建first_stage和second_stage这两个子文件夹
    
    # 加载数据集
    train_dataloader = DataLoader(database, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_database = build_dataset(args.dataset, args.data_dir, channel=args.input_nc, isTraining=False,
                                 crop_size=(args.crop_size, args.crop_size), scale_size=(args.scale_size, args.scale_size))
    val_dataloader = DataLoader(val_database, batch_size=1)
    
    # 构建模型
    # first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1).to(device)
    # first_net = torch.nn.DataParallel(first_net)
    # first_optim = optim.Adam(first_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    first_net_thick = torch.load(args.models_dir + "/" + args.dataset + "/first_stage/front_model-" + args.first_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
    first_net_thick = first_net_thick.module
    first_net_thick.eval()
    first_net_thin = torch.load(args.models_dir + "/" + args.dataset + "/first_stage/front_model-" + args.first_suffix1).to(device)  # two stage时可以加载first_stage和second_stage的模型
    first_net_thin = first_net_thin.module
    first_net_thin.eval()
    second_net = fusion(channels=args.base_channels, pn_size=args.pn_size, kernel_size=3, avg=0.0, std=0.1).to(device)  # ##
    second_net = torch.nn.DataParallel(second_net)  # ##
    # second_net = torch.load(args.models_dir + "/" + args.dataset + "/second_stage_v2/fusion_model-best_fusion.pth").to(device)
    # second_net = second_net
    second_optim = optim.Adam(second_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    
    thick_criterion = torch.nn.MSELoss()  # 可更改
    thin_criterion = torch.nn.MSELoss()  # 可更改
    fusion_criterion = torch.nn.MSELoss()  # 可更改
    
    best_fusion = {"epoch": 217, "auc": 0.86}
    # start training
    print("Start training...")
    for epoch in range(args.second_epochs):
        print('Epoch %d / %d' % (epoch + 218, args.second_epochs + 217))
        print('-'*10)
        second_net = train_second_stage(viz, writer, train_dataloader, first_net_thick, first_net_thin, second_net, second_optim, args.init_lr, fusion_criterion, device, args.power, epoch+300, args.second_epochs+300)
        if (epoch + 1) % args.val_epoch_freq == 0 or epoch == args.second_epochs - 1:
            second_net, best_fusion = val_second_stage(best_fusion, viz, writer, val_dataloader,first_net_thick, first_net_thin, second_net,
                                                       fusion_criterion, device, args.save_epoch_freq,
                                                       args.models_dir + "/" + sub_dir, args.results_dir + "/" + sub_dir, epoch+800, args.second_epochs+800)
    print("Training finished.")
else:  # test
    # 加载数据集和模型
    test_dataloader = DataLoader(database, batch_size=1)
    first_net = torch.load(args.models_dir + "/" + args.dataset + "/first_stage/front_model-" + args.first_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
    first_net = first_net.module
    first_net.eval()
    first_net1 = torch.load(args.models_dir + "/" + args.dataset + "/first_stage/front_model-" + args.first_suffix1).to(device)  # two stage时可以加载first_stage和second_stage的模型
    first_net1 = first_net1.module
    first_net1.eval()
    second_net = torch.load(args.models_dir + "/" + sub_dir + "/fusion_model-" + args.second_suffix).to(device)  # two stage时可以加载first_stage和second_stage的模型
    second_net.eval()
    
    # start testing
    print("Start testing...")
    test_second_stage(test_dataloader, first_net,  first_net1, second_net, device, args.results_dir + "/" + sub_dir, criterion=None, isSave=True)
    print("Testing finished.")
