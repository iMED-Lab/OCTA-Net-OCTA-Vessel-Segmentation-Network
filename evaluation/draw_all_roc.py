# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


bwith = 0.4
fig = plt.figure(figsize=(24, 8))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

ax = fig.add_subplot(1, 2, 1)
# ax = plt.gca()
ax.set_title("ROSE-1", fontsize=20)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.grid(linestyle='-.',linewidth=0.05)


rosea_method_lst = ["IPAC", "COSFIRE", "COOF", "U-Net", "ResU-Net", "CE-Net", "DUNet", "CS-Net", "three-stage", "Backbone (ResNeSt)", "Backbone (jointly learning)", "Ours"]
rosea_color_lst = ["y", "b", "g", "r", "y", "b", "g", "r", "y", "b", "g", "r"]
rosea_marker_lst = ["+", "x", "o", "d", "*", "s", "+", "x", "o", "d", "*", "s"]
rosea_linestyle_lst = ["-.", "--", "-", "-.", "--", "-", "-.", "--", "-", "-.", "--", "-"]
assert len(rosea_method_lst) == len(rosea_color_lst) and len(rosea_color_lst) == len(rosea_linestyle_lst)

rosea_zip = list(zip(rosea_color_lst, rosea_marker_lst, rosea_linestyle_lst))
rosea_dct = dict(zip(rosea_method_lst, rosea_zip))
gt_lst = sorted(os.listdir("ROSE-A_results/all/gt"))
gt_arr_lst = []
for gt in gt_lst:
    gt_arr_lst.append(cv2.imread("ROSE-A_results/all/gt/" + gt, 0) // 255)
gt_vec = np.stack(gt_arr_lst, axis=0).reshape(-1)


for rosea_method in rosea_method_lst:
    prob_lst = sorted(os.listdir("ROSE-A_results/all/" + rosea_method + "/prob"))
    assert len(prob_lst) == len(gt_lst)
    
    prob_arr_lst = []
    for i in range(len(gt_lst)):
        prob_arr_lst.append(cv2.imread("ROSE-A_results/all/" + rosea_method + "/prob/" + prob_lst[i], 0) / 255.0)
    
    prob_vec = np.stack(prob_arr_lst, axis=0).reshape(-1)
    fpr, tpr, thresholds = metrics.roc_curve(gt_vec, prob_vec, pos_label=1)
    roc_auc = metrics.roc_auc_score(gt_vec, prob_vec)
    plt.plot(fpr, tpr, label=rosea_method+" (AUC={0:.4f})".format(roc_auc),
             color=rosea_dct[rosea_method][0], marker=rosea_dct[rosea_method][1], linestyle=rosea_dct[rosea_method][2],
             linewidth=0.7, markersize=4)


font = {'family': 'Liberation Sans',
        'weight': 'normal',
        'size': 14}
font1 = {'family': 'Liberation Sans',
         'weight': 'normal',
         'size': 20}
plt.xlim(0, 1)
plt.ylim(0.5, 1)
plt.xticks(np.linspace(0, 1, 6))
plt.yticks(np.linspace(0.5, 1, 6))
plt.xlabel('1-Specificity',font1)
plt.ylabel('Sensitivity',font1)
plt.xticks(fontproperties='Liberation Sans', size=12, weight='normal')
plt.yticks(fontproperties='Liberation Sans', size=12, weight='normal')

plt.legend(loc='lower right',prop=font)
# plt.savefig('./figures/rosea.eps')
# plt.savefig('./figures/rosea.png')
# plt.show()

# ##################################################################################################

ax = fig.add_subplot(1, 2, 2)
# ax = plt.gca()
ax.set_title("ROSE-2", fontsize=20)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.grid(linestyle='-.',linewidth=0.05)


roseb_method_lst = ["IPAC", "COSFIRE", "COOF", "U-Net", "ResU-Net", "CE-Net", "DUNet", "CS-Net", "three-stage", "Backbone (ResNeSt)", "Ours"]
roseb_color_lst = ["lightcoral", "yellow", "cyan", "red", "lime", "purple", "orange", "gray", "blue", "green", "deeppink"]
roseb_marker_lst = ["+", "x", "o", "d", "*", "s", "+", "x", "o", "d", "s"]
roseb_linestyle_lst = ["-.", "--", "-", "-.", "--", "-", "-.", "--", "-", "-.", "-"]
assert len(roseb_method_lst) == len(roseb_color_lst) and len(roseb_color_lst) == len(roseb_linestyle_lst)

roseb_zip = list(zip(roseb_color_lst, roseb_marker_lst, roseb_linestyle_lst))
roseb_dct = dict(zip(roseb_method_lst, roseb_zip))
gt_lst = sorted(os.listdir("ROSE-B_opt_results/gt"))
gt_arr_lst = []
for gt in gt_lst:
    gt_arr_lst.append(cv2.imread("ROSE-B_opt_results/gt/" + gt, 0) // 255)
gt_vec = np.stack(gt_arr_lst, axis=0).reshape(-1)


for roseb_method in roseb_method_lst:
    prob_lst = sorted(os.listdir("ROSE-B_opt_results/" + roseb_method + "/prob"))
    assert len(prob_lst) == len(gt_lst)
    
    prob_arr_lst = []
    for i in range(len(gt_lst)):
        prob_arr_lst.append(cv2.imread("ROSE-B_opt_results/" + roseb_method + "/prob/" + prob_lst[i], 0) / 255.0)
    
    prob_vec = np.stack(prob_arr_lst, axis=0).reshape(-1)
    fpr, tpr, thresholds = metrics.roc_curve(gt_vec, prob_vec, pos_label=1)
    roc_auc = metrics.roc_auc_score(gt_vec, prob_vec)
    plt.plot(fpr, tpr, label=roseb_method+" (AUC={0:.4f})".format(roc_auc),
             color=roseb_dct[roseb_method][0], marker=roseb_dct[roseb_method][1], linestyle=roseb_dct[roseb_method][2],
             linewidth=0.7, markersize=4)


font = {'family': 'Liberation Sans',
        'weight': 'normal',
        'size': 14}
font1 = {'family': 'Liberation Sans',
         'weight': 'normal',
         'size': 20}
plt.xlim(0, 1)
plt.ylim(0.75, 1)
plt.xticks(np.linspace(0, 1, 6))
plt.yticks(np.linspace(0.75, 1, 6))
plt.xlabel('1-Specificity',font1)
plt.ylabel('Sensitivity',font1)
plt.xticks(fontproperties='Liberation Sans', size=12, weight='normal')
plt.yticks(fontproperties='Liberation Sans', size=12, weight='normal')

plt.legend(loc='lower right',prop=font)
plt.savefig('./rose.eps')  # figures/
plt.savefig('./rose.png')  # figures/
plt.show()
