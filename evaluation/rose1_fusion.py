# -*- coding: utf-8 -*-

import os
import cv2
import argparse
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument("--superficial_dir", type=str, required=True, help="path to folder for getting superficial maps")
parser.add_argument("--deep_dir", type=str, required=True, help="path to folder for getting deep maps")
parser.add_argument("--all_dir", type=str, required=True, help="path to folder for saving fusion maps")
args = parser.parse_args()

superficial_prob_lst = sorted(os.listdir(os.path.join(args.superficial_dir, "prob")))
deep_prob_lst = sorted(os.listdir(os.path.join(args.deep_dir, "prob")))

superficial_pred_lst = sorted(os.listdir(os.path.join(args.superficial_dir, "pred")))
deep_pred_lst = sorted(os.listdir(os.path.join(args.deep_dir, "pred")))

assert len(superficial_prob_lst) == len(deep_prob_lst)
assert len(superficial_pred_lst) == len(deep_pred_lst)
assert len(deep_prob_lst) == len(deep_pred_lst)

for i in range(len(superficial_prob_lst)):
    superficial_prob_arr = cv2.imread(os.path.join(args.superficial_dir, "prob", superficial_prob_lst[i]), 0)
    deep_prob_arr = cv2.imread(os.path.join(args.deep_dir, "prob", deep_prob_lst[i]), 0)
    all_prob_arr = max_fusion(superficial_prob_arr, deep_prob_arr)
    cv2.imwrite(os.path.join(args.all_dir, "prob", superficial_prob_lst[i]), all_prob_arr)
    
    superficial_pred_arr = cv2.imread(os.path.join(args.superficial_dir, "pred", superficial_pred_lst[i]), 0)
    deep_pred_arr = cv2.imread(os.path.join(args.deep_dir, "pred", deep_pred_lst[i]), 0)
    all_pred_arr = max_fusion(superficial_pred_arr, deep_pred_arr)
    cv2.imwrite(os.path.join(args.all_dir, "pred", superficial_pred_lst[i]), all_pred_arr)
    
    print(superficial_prob_lst[i] + " done.")
