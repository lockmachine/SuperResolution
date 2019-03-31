#!/usr/bin/env python3
# coding:utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict
from common.util import im2col, col2im
from common.optimizer import *
from common.layers import *
from srcnn import SRCNN
import cv2
import glob
from tqdm import tqdm
import time
import pickle
from visualize_filter import filter_show

# 訓練データの読み込み
x_test_path = './Test/Train_LR/*'
t_test_path = './Test/Train_HR/*'
x_test_pkl  = './Test/Train_LR.pkl'
x_test_list = glob.glob(x_test_path)
t_test_list = glob.glob(t_test_path)

x_test_img_list = None
t_test_img_list = None

use_epochs = np.arange(200, 201, 100)
idx = 0
for use_epoch in use_epochs: 
    for img_file in x_test_list:
        img = cv2.imread(img_file)  # 縦 x 横 x チャンネル(BGR形式)
        print(img_file)
        fname = os.path.basename(img_file)
        ftitle, fext = os.path.splitext(fname)  # baby_GT_LR, .bmp
        idx += 1

        #with open(x_test_pkl, "wb") as f:
        #    pickle.dump(x_test_img_list, f)

        """
        # pickle から訓練データと正解データの読み込み
        with open(x_test_pkl, "rb") as f:
            x_test_img_list = pickle.load(f)
        """
        #print(img.shape)

        # Set5データを整形 (X, Y, C) → (C, X, Y)に変更
        img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])
        img = img.transpose(0, 3, 1, 2)
        #print(img.shape)

        input_dim=img.shape

        # ネットワーク初期化後の重みの読み込み
        param_file = './train_proc/param_' + str(use_epoch) + 'epochs.pkl'
        with open(param_file, "rb") as f:
            params = pickle.load(f)

        #filter_show(params["W1"])
        #filter_show(params["W2"])
        #filter_show(params["W3"])

        # ネットワークにSRCNNを設定
        network = SRCNN(input_dim=input_dim,
                        fUseParam=True,
                        W1=params['W1'],
                        W2=params['W2'],
                        W3=params['W3'],
                        b1=params['b1'],
                        b2=params['b2'],
                        b3=params['b3'])


        y = network.predict(img)

        y = y.reshape(y.shape[1], y.shape[2], y.shape[3])
        y = y.transpose(1, 2, 0)

        y = y - np.min(y)
        y = (y / np.max(y)) * 255

        #print(y.shape)
        #print(np.max(y))
        #print(np.min(y))
        y = y.astype('uint8')
        #print(y[0])
        #print(type(y))
        #cv2.imshow('y', y)
        #cv2.waitKey(0)
        write_img_file_name = './Test/Train_HR/' + ftitle + '_HR_'+ str(use_epoch) + 'epoch.bmp'
        cv2.imwrite(write_img_file_name, y)
        #filter_show(y)
