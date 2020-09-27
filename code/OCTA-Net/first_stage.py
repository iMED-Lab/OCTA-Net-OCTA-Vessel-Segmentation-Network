# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import resnest
from splat import SplAtConv2d

# #########--------- Components ---------#########
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            SplAtConv2d(ch_out, ch_out, kernel_size=3,padding=1,groups=2,radix=2,norm_layer=nn.BatchNorm2d),
            nn.ReLU(inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        
        return self.relu(out + residual)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.up(x)
        
        return x


# #########--------- Networks ---------#########
class SRF_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(SRF_UNet, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnest.resnest50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.Up5_thick = up_conv(ch_in=2048, ch_out=1024)
        self.Up_conv5_thick = res_conv_block(ch_in=2048, ch_out=1024)
        
        self.Up4_thick = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv4_thick = res_conv_block(ch_in=1024, ch_out=512)
        
        self.Up3_thick = up_conv(ch_in=512, ch_out=256)
        self.Up_conv3_thick = res_conv_block(ch_in=512, ch_out=256)
        
        self.Up2_thick = up_conv(ch_in=256, ch_out=64)
        self.Up_conv2_thick = res_conv_block(ch_in=128, ch_out=64)
        
        self.Up1_thick = up_conv(ch_in=64, ch_out=64)
        self.Up_conv1_thick = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thick = nn.Conv2d(32, output_ch, kernel_size=1)
        
        # ##
        self.Up5_thin = up_conv(ch_in=2048, ch_out=1024)
        self.Up_conv5_thin = res_conv_block(ch_in=2048, ch_out=1024)
        
        self.Up4_thin = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv4_thin = res_conv_block(ch_in=1024, ch_out=512)
        
        self.Up3_thin = up_conv(ch_in=512, ch_out=256)
        self.Up_conv3_thin = res_conv_block(ch_in=512, ch_out=256)
        # ##
        
        self.Up2_thin = up_conv(ch_in=256, ch_out=64)
        self.Up_conv2_thin = res_conv_block(ch_in=128, ch_out=64)
        
        self.Up1_thin = up_conv(ch_in=64, ch_out=64)
        self.Up_conv1_thin = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thin = nn.Conv2d(32, output_ch, kernel_size=1)
        
        # ##
        self.Up_conv1 = res_conv_block(ch_in=64,ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1)
        # ##
    
    def forward(self, x):
        # encoding path
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)
        
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        
        down_pad = False
        right_pad = False
        if x4.size()[2] % 2 == 1:
            x4 = F.pad(x4, (0, 0, 0, 1))
            down_pad = True
        if x4.size()[3] % 2 == 1:
            x4 = F.pad(x4, (0, 1, 0, 0))
            right_pad = True
        
        x5 = self.encoder4(x4)
        # print(x0.size(),x1.size(),x2.size(),x3.size(),x4.size(),)
        
        # decoding + concat path
        d5_thick = self.Up5_thick(x5)
        d5_thick = torch.cat((x4, d5_thick), dim=1)
        
        # Decoder
        if down_pad and (not right_pad):
            d5_thick = d5_thick[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5_thick = d5_thick[:, :, :, :-1]
        if down_pad and right_pad:
            d5_thick = d5_thick[:, :, :-1, :-1]
        
        d5_thick = self.Up_conv5_thick(d5_thick)
        
        d4_thick = self.Up4_thick(d5_thick)
        d4_thick = torch.cat((x3, d4_thick), dim=1)
        d4_thick = self.Up_conv4_thick(d4_thick)
        
        d3_thick = self.Up3_thick(d4_thick)
        d3_thick = torch.cat((x2, d3_thick), dim=1)
        d3_thick = self.Up_conv3_thick(d3_thick)
        
        d2_thick = self.Up2_thick(d3_thick)
        d2_thick = torch.cat((x0, d2_thick), dim=1)
        d2_thick = self.Up_conv2_thick(d2_thick)
        
        d1_thick = self.Up1_thick(d2_thick)
        # d1_thick = torch.cat((x, d1_thick), dim=1)
        d1_thick = self.Up_conv1_thick(d1_thick)
        
        d1_thick = self.Conv_1x1_thick(d1_thick)
        out_thick = nn.Sigmoid()(d1_thick)
        
        """
        d5_thin = self.Up5_thin(x5)
        d5_thin = torch.cat((x4, d5_thin), dim=1)
        
        d5_thin = self.Up_conv5_thin(d5_thin)
        
        d4_thin = self.Up4_thin(d5_thin)
        d4_thin = torch.cat((x3, d4_thin), dim=1)
        d4_thin = self.Up_conv4_thin(d4_thin)
        
        d3_thin = self.Up3_thin(d4_thin)  # x3
        d3_thin = torch.cat((x2, d3_thin), dim=1)
        d3_thin = self.Up_conv3_thin(d3_thin)
        """
        
        d2_thin = self.Up2_thin(x2)  # d3_thin
        d2_thin = torch.cat((x0, d2_thin), dim=1)
        d2_thin = self.Up_conv2_thin(d2_thin)
        
        d1_thin = self.Up1_thin(d2_thin)
        # d1_thin = torch.cat((x, d1_thin), dim=1)
        d1_thin = self.Up_conv1_thin(d1_thin)
        
        d1_thin = self.Conv_1x1_thin(d1_thin)
        out_thin = nn.Sigmoid()(d1_thin)
        
        assert out_thick.size() == out_thin.size()
        out = torch.max(out_thick, out_thin)
        """
        d1 = torch.cat([out_thick, out_thin], dim=1)  # d2_thick, d2_thin
        d1 = self.Up_conv1(d1)
        out = self.Conv_1x1(d1)
        out = nn.Sigmoid()(out)
        """
        
        return out_thick, out_thin, out
