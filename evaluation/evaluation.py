# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--prob_dir", type=str, required=True, help="path to folder for saving probability maps")
parser.add_argument("--pred_dir", type=str, required=True, help="path to folder for saving prediction maps")
parser.add_argument("--gt_dir", type=str, required=True, help="path to folder for saving ground truth")
args = parser.parse_args()

prob_lst = sorted(os.listdir(args.prob_dir))
pred_lst = sorted(os.listdir(args.pred_dir))
gt_lst = sorted(os.listdir(args.gt_dir))
assert len(prob_lst) == len(pred_lst) and len(pred_lst) == len(gt_lst)

metric_dct = {"auc": [], "acc": [], "sen": [], "spe": [], "gmean": [], "kappa": [], "fdr": [], "iou": [], "dice": []}
for i in range(len(pred_lst)):
    prob_arr = cv2.imread(os.path.join(args.prob_dir, prob_lst[i]), 0) / 255.0
    pred_arr = cv2.imread(os.path.join(args.pred_dir, pred_lst[i]), 0) // 255
    gt_arr = cv2.imread(os.path.join(args.gt_dir, gt_lst[i]), 0) // 255
    
    metric_dct["auc"].append(calc_auc(prob_arr, gt_arr))
    metric_dct["acc"].append(calc_acc(pred_arr, gt_arr))
    metric_dct["sen"].append(calc_sen(pred_arr, gt_arr))
    metric_dct["spe"].append(calc_spe(pred_arr, gt_arr))
    metric_dct["gmean"].append(calc_gmean(pred_arr, gt_arr))
    metric_dct["kappa"].append(calc_kappa(pred_arr, gt_arr))
    metric_dct["fdr"].append(calc_fdr(pred_arr, gt_arr))
    metric_dct["iou"].append(calc_iou(pred_arr, gt_arr))
    metric_dct["dice"].append(calc_dice(pred_arr, gt_arr))

for key, value in metric_dct.items():
    print(key + "\tmean: " + str(np.array(value).mean()) + "\tstd: " + str(np.array(value).std()))
