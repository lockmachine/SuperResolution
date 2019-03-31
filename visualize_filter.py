#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from srcnn import SRCNN
import pickle

def filter_show(filters, nx=8, margin=3, scale=10):
    FN, C, FH, FW = filters.shape
    
    # プロットの行数
    ny = int(np.ceil(FN / nx))
    
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()

if __name__ == '__main__':
    network = SRCNN()

    # ネットワーク初期化後の重み
    param_file = './train_proc/param_0epochs.pkl'
    with open(param_file, "rb") as f:
        params = pickle.load(f)

    filter_show(params["W1"])
    filter_show(params["W2"])
    filter_show(params["W3"])


    # 学習後の重み
    param_file = './train_proc/param_200epochs.pkl'
    with open(param_file, "rb") as f:
        params = pickle.load(f)

    filter_show(params["W1"])
    filter_show(params["W2"])
    filter_show(params["W3"])
