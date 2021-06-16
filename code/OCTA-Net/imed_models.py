# -*- coding: utf-8 -*-

"""
models from imed:
Context Encoder Network (CE-Net, TMI 2019)
Channel and Spatial CSNet Network (CS-Net, MICCAI 2019).
"""
from __future__ import division
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)


# #########--------- CE-Net ---------#########
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
    
    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        
        return x


class CE_Net(nn.Module):
    def __init__(self, num_classes=1):
        super(CE_Net, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = DACblock(512)
        self.spp = SPPblock(512)
        
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
    
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        down_pad = False
        right_pad = False
        if e3.size()[2] % 2 == 1:
            e3 = F.pad(e3, (0, 0, 0, 1))
            down_pad = True
        if e3.size()[3] % 2 == 1:
            e3 = F.pad(e3, (0, 1, 0, 0))
            right_pad = True
        
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)
        
        # Decoder
        if (not down_pad) and (not right_pad):
            d4 = self.decoder4(e4) + e3
        elif down_pad and (not right_pad):
            d4 = self.decoder4(e4)[:, :, :-1, :] + e3[:, :, :-1, :]
        elif (not down_pad) and right_pad:
            d4 = self.decoder4(e4)[:, :, :, :-1] + e3[:, :, :, :-1]
        else:
            d4 = self.decoder4(e4)[:, :, :-1, :-1] + e3[:, :, :-1, :-1]
        
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        
        return F.sigmoid(out)


# #########--------- CS-Net ---------#########
def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out += residual
        out = self.relu(out)
        
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.conv(x)
        
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1))
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0))
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        :param x: input( B x C x H x W )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        :param x: input( B x C x H x W )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """
    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
    
    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        
        return out


class CS_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        :param out_channels: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CS_Net, self).__init__()
        self.enc_input = ResEncoder(in_channels, 64)  # 32##
        self.encoder1 = ResEncoder(64, 128)  # (32, 64)##
        self.encoder2 = ResEncoder(128, 256)  # (64, 128)##
        self.encoder3 = ResEncoder(256, 512)  # (128, 256)##
        self.encoder4 = ResEncoder(512, 1024)  # (256, 512)##
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention(1024)  # 512##
        self.attention_fuse = nn.Conv2d(1024 * 2, 1024, kernel_size=1)  # 512##
        self.decoder4 = Decoder(1024, 512)  # (512, 256)##
        self.decoder3 = Decoder(512, 256)  # (256, 128)##
        self.decoder2 = Decoder(256, 128)  # (128, 64)##
        self.decoder1 = Decoder(128, 64)  # (64, 32)##
        self.deconv4 = deconv(1024, 512)  # (512, 256)##
        self.deconv3 = deconv(512, 256)  # (256, 128)##
        self.deconv2 = deconv(256, 128)  # (128, 64)##
        self.deconv1 = deconv(128, 64)  # (64, 32)##
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # 32##
        initialize_weights(self)
    
    def forward(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)
        
        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)
        
        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)
        
        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)
        
        input_feature = self.encoder4(down4)
        
        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)
        
        # attention_fuse = self.attention_fuse(torch.cat((input_feature, attention), dim=1))
        attention_fuse = input_feature + attention
        
        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)
        
        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)
        
        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)
        
        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)
        
        final = self.final(dec1)
        final = F.sigmoid(final)
        
        return final
