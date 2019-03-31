#!/usr/bin/env python3
# coding:utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ネットワーク初期化後の重みの読み込み
acc_file = './train_proc/acc_list.pkl'
with open(acc_file, "rb") as f:
    acc_list = pickle.load(f)

loss_file = './train_proc/loss_list.pkl'
with open(loss_file, "rb") as f:
    loss_list = pickle.load(f)

plt.plot(loss_list)
plt.show()
