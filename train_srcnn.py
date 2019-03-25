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

# 訓練データの読み込み
x_train_path = './Train_LR_split/*'
t_train_path = './Train_HR_split/*'
x_train_list = glob.glob(x_train_path)
t_train_list = glob.glob(t_train_path)

x_train_img_list = None
t_train_img_list = None

"""
pbar = tqdm(x_train_list)
pbar.set_description("Read x_train")
idx = 0
for img_file in pbar:
    img = cv2.imread(img_file)  # 縦 x 横 x チャンネル(BGR形式)
    img = img.reshape(-1, 33, 33, 3)
    if idx == 0:
        x_train_img_list = img
    else:
        x_train_img_list = np.append(x_train_img_list, img, axis=0)
    idx += 1

with open('LR_split_data.pkl', "wb") as f:
    pickle.dump(x_train_img_list, f)
"""

# pickle から訓練データと正解データの読み込み
with open('LR_split_data.pkl', "rb") as f:
    x_train_img_list = pickle.load(f)


"""
pbar = tqdm(t_train_list)
pbar.set_description("Read t_train")
idx = 0
for img_file in pbar:
    img = cv2.imread(img_file)  # 縦 x 横 x チャンネル(BGR形式)
    img = img.reshape(-1, 33, 33, 3)
    # パディングがない場合は 21x21 のサイズになるのであらかじめ削っておく
    img = img[:, 6:27, 6:27, :]
    if idx == 0:
        t_train_img_list = img
    else:
        t_train_img_list = np.append(t_train_img_list, img, axis=0)
    idx += 1

with open('HR_split_data.pkl', "wb") as f:
    pickle.dump(t_train_img_list, f)

"""
with open('HR_split_data.pkl', "rb") as f:
    t_train_img_list = pickle.load(f)

# (4364, 33, 33, 3) → (4364, 3, 33, 33)に変更
x_train_img_list = x_train_img_list.transpose(0, 3, 1, 2)
t_train_img_list = t_train_img_list.transpose(0, 3, 1, 2)
#print(x_train_img_list.shape)
#print(t_train_img_list.shape)


# パラメータの設定
batch_size = 100
iter_num = 0
max_iter_num = 10000
max_epoch_num = 20

# 1エポックの訓練回数
train_per_epoch = x_train_img_list.shape[0] // batch_size + 1

train_loss_list = []
train_acc_list = []

# ネットワークにSRCNNを設定
network = SRCNN()

# パラメータ更新関数にMomentumを設定
optimizer = Momentum()

start = time.time()
epoch_num = 0
loss = 0
acc = 0
for i in range(max_iter_num):
    batch_mask = np.random.choice(x_train_img_list.shape[0], batch_size)
    
    # バッチデータの取り出し
    x_train_batch = x_train_img_list[batch_mask]/255    # (100, 3, 33, 33), <class 'numpy.float64'>
    t_train_batch = t_train_img_list[batch_mask]/255    # (100, 3, 21, 21), <class 'numpy.float64'>
    
    # 勾配計算
    grad = network.gradient(x_train_batch, t_train_batch)
    
    """
    print('grad W1')
    print(grad['W1'][0, 0, 0])
    print('grad b1')
    print(grad['b1'][0])
    print('grad W2')
    print(grad['W2'][0, 0, 0])
    print('grad b2')
    print(grad['b2'][0])
    print('grad W3')
    print(grad['W3'][0, 0, 0])
    print('grad b3')
    print(grad['b3'][0])
    """
    
    # パラメータの更新
    optimizer.update(network.params, grad)
    
    #print('W1')
    #print(network.params['W1'][0, 0, 0])
    
    
    # 損失関数計算
    loss = network.loss(x_train_batch, t_train_batch)
    train_loss_list.append(loss)
    
    # 再現精度計算
    acc = network.accuracy(x_train_batch, t_train_batch)
    train_acc_list.append(acc)
    
    print('{} iter train_loss:{}  train_acc :{}'.format(i, loss, acc))

    if i % train_per_epoch == 0:
        epoch_num += 1
        end = time.time()
        print('------{}/{} epochs(TotalProcTime:{})------'.format(epoch_num, max_epoch_num, end-start))
        print('train_loss:{}\ntrain_acc :{}'.format(loss, acc))
        if epoch_num > max_epoch_num:
            break



# パラメータの保存
#network.save_params('params.pkl')
