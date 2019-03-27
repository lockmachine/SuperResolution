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
x_test_path = './Test/Set5_LR/*'
t_test_path = './Test/Set5_HR/*'
x_test_pkl  = './Test/Set5_LR.pkl'
x_test_list = glob.glob(x_test_path)
t_test_list = glob.glob(t_test_path)

x_test_img_list = None
t_test_img_list = None


idx = 0
for img_file in x_test_list:
    img = cv2.imread(img_file)  # 縦 x 横 x チャンネル(BGR形式)
    idx += 1
    break

#with open(x_test_pkl, "wb") as f:
#    pickle.dump(x_test_img_list, f)

"""
# pickle から訓練データと正解データの読み込み
with open(x_test_pkl, "rb") as f:
    x_test_img_list = pickle.load(f)
"""
print(img.shape)

# Set5データを整形 (X, Y, C) → (C, X, Y)に変更
img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])
img = img.transpose(0, 3, 1, 2)
print(img.shape)

input_dim=img.shape

# ネットワーク初期化後の重みの読み込み
param_file = './train_proc/param_50epochs.pkl'
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

print(y.shape)
print(np.max(y))
print(np.min(y))
y = y.astype('uint8')
print(y[0])
print(type(y))
cv2.imshow('y', y)
cv2.imwrite('./Test/Set5_HR/baby50.bmp', y)
cv2.waitKey(0)
#filter_show(y)

exit()

# 学習後の重み
param_file = './train_proc/param_50epochs.pkl'
with open(param_file, "rb") as f:
    params_a = pickle.load(f)


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
        
        param_file = './train_proc/param_'+str(epoch_num - 1)+'epochs.pkl'
        with open(param_file, "wb") as f:
            pickle.dump(network.params, f)
        
        loss_file = './train_proc/loss_'+str(epoch_num - 1)+'epochs.pkl'
        with open(loss_file, "wb") as f:
            pickle.dump(train_loss_list, f)
        
        acc_file = './train_proc/acc_'+str(epoch_num - 1)+'epochs.pkl'
        with open(acc_file, "wb") as f:
            pickle.dump(train_acc_list, f)
            
        if epoch_num > max_epoch_num:
            break

plt.plot(train_loss_list)
plt.plot(train_acc_list)
plt.xlabel('iter')
plt.ylabel('loss & accuracy')
plt.show()
# パラメータの保存
#network.save_params('params.pkl')
