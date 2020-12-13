# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from splat import SplAtConv2d
import resnest
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


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            SplAtConv2d(ch_out, ch_out, kernel_size=3, padding=1, groups=4, radix=4, norm_layer=nn.BatchNorm2d),
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


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return x * psi


# #########--------- Networks ---------#########
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)
        
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)
    
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out


# #########--------- Networks ---------#########


class DAB(nn.Module):
    def __init__(self, reduction=16):
        super(DAB, self).__init__()
        self.reduction = reduction
        self.relu = nn.ReLU(inplace=False)

    def forward(self, fea_low, fea_high):
        assert fea_low.size() == fea_high.size()
        b, c, h, w = fea_low.size()
        sq = nn.Linear(2 * c, 2 * c // self.reduction, bias=True).cuda()
        ex = nn.Linear(2 * c // self.reduction, c, bias=True).cuda()

        fea_com = fea_high + fea_low  # b, c, h, w
        fea_diff = fea_high - fea_low  # b, c, h, w
        feas = torch.cat([fea_high, fea_low], dim=1)  # b, 2 * c, h, w
        fea_sq = self.relu(sq(feas.mean(-1).mean(-1)))  # b, 2 * c // self.reduction
        w_diff = F.sigmoid(ex(fea_sq))  # b, c
        w_com = 1.0 - w_diff  # b, c

        final_com = fea_com * w_com.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(fea_com)  # b, c, h, w
        final_diff = fea_diff * w_diff.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(fea_diff)  # b, c, h, w
        final_out = torch.cat((final_com,final_diff),dim=1)  # b, c, h, w

        return final_out


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

        #self.dab1 = DAB()
        #self.dab2 = DAB()
        #self.dab3 = DAB()
        #self.dab4 = DAB()
        #self.Up5_thick = up_conv(ch_in=2048, ch_out=1024)
        #self.Up_conv5_thick = res_conv_block(ch_in=2048, ch_out=1024)

        self.Up4_thick = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv4_thick = res_conv_block(ch_in=1024, ch_out=512)

        self.Up3_thick = up_conv(ch_in=512, ch_out=256)
        self.Up_conv3_thick = res_conv_block(ch_in=512, ch_out=256)

        self.Up2_thick = up_conv(ch_in=256, ch_out=64)
        self.Up_conv2_thick = res_conv_block(ch_in=128, ch_out=64)

        self.Up1_thick = up_conv(ch_in=64, ch_out=64)
        self.Up_conv1_thick = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thick = nn.Conv2d(32, output_ch, kernel_size=1)
        """
        self.Up5_thin = up_conv(ch_in=2048, ch_out=1024)
        self.Up_conv5_thin = res_conv_block(ch_in=2048, ch_out=1024)

        self.Up4_thin = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv4_thin = res_conv_block(ch_in=1024, ch_out=512)

        self.Up3_thin = up_conv(ch_in=512, ch_out=256)
        self.Up_conv3_thin = res_conv_block(ch_in=512, ch_out=256)

        self.Up2_thin = up_conv(ch_in=256, ch_out=64)
        self.Up_conv2_thin = res_conv_block(ch_in=128, ch_out=64)

        self.Up1_thin = up_conv(ch_in=64, ch_out=64)
        self.Up_conv1_thin = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1_thin = nn.Conv2d(32, output_ch, kernel_size=1)
        self.Up_conv1 = res_conv_block(ch_in=64,ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1)
        """

    def forward(self, x):
        # encoding path
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        # down_pad = False
        # right_pad = False
        # if x4.size()[2]%2==1:
        #    x4 = F.pad(x4,(0,0,0,1))
        #    down_pad = True
        # if x4.size()[3]%2==1:
        #    x4 = F.pad(x4,(0,1,0,0))
        #    right_pad = True
        # x5 = self.encoder4(x4)
        #
        # # print(x0.size(),x1.size(),x2.size(),x3.size(),x4.size(),)
        # # decoding + concat path
        # d5 = self.Up5_thick(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        # if down_pad and (not right_pad):
        #     d5 = d5[:,:,:-1,:]
        # if (not down_pad) and right_pad:
        #     d5 = d5[:,:,:,:-1]
        # if down_pad and right_pad:
        #    d5 = d5[:,:,:-1,:-1]
        # #
        # #
        # d5 = self.Up_conv5_thick(d5)

        d4 = self.Up4_thick(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4_thick(d4)

        d3 = self.Up3_thick(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3_thick(d3)

        d2 = self.Up2_thick(d3)
        d2 = torch.cat((x0, d2), dim=1)
        d2 = self.Up_conv2_thick(d2)

        d2 = self.Up1_thick(d2)
        d2 = self.Up_conv1_thick(d2)

        d1 = self.Conv_1x1_thick(d2)
        out = nn.Sigmoid()(d1)

        return out


class ResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(ResUNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = res_conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = res_conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = res_conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=64)
        self.Up_conv2 = res_conv_block(ch_in=128, ch_out=64)

        self.Up1 = up_conv(ch_in=64, ch_out=64)
        self.Up_conv1 = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        # decoding + concat path
        #d5 = self.Up5(x5)
        #d5 = torch.cat([x4, d5],dim=1)

        #d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat([x3, d4],dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat([x2, d3],dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat([x0, d2],dim=1)
        d2 = self.Up_conv2(d2)

        d2 = self.Up1(d2)
        d2 = self.Up_conv1(d2)

        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)

        return out


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        
        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out


class AttResU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttResU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = res_conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = res_conv_block(ch_in=64, ch_out=128)
        self.Conv3 = res_conv_block(ch_in=128, ch_out=256)
        self.Conv4 = res_conv_block(ch_in=256, ch_out=512)
        self.Conv5 = res_conv_block(ch_in=512, ch_out=1024)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = res_conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = res_conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = res_conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = res_conv_block(ch_in=128, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out
