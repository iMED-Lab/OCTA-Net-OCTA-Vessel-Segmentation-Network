# -*- coding: utf-8 -*-

import os
import time
import torch
from options import args
from first_stage import SRF_UNet
from second_stage import fusion


# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.randn(1, args.input_nc, 512, 512).to(device)
# first_net = SRF_UNet(img_ch=args.input_nc, output_ch=1).to(device)
second_net = fusion(channels=32, pn_size=3).to(device)
start_t = time.time()
outputs = second_net(inputs, inputs, inputs)
end_t = time.time()
print("cost time: %f s" % (end_t - start_t))
