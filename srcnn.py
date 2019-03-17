#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict
from common.util import im2col, col2im
from common.optimizer import *
from common.layers import *

im = cv2.imread('Train/t1.bmp') # BGR形式
cv2.imshow('t1.bmp',im[:,:,0])  # B
cv2.waitKey(0)
cv2.imshow('t1.bmp',im[:,:,1])  # G
cv2.waitKey(0)
cv2.imshow('t1.bmp',im[:,:,2])  # R
print(im.shape) # ex) 176 x 197 x 3
cv2.waitKey(0)
cv2.destroyAllWindows()

class SRCNN:
    """
    input_dim:入力画像サイズ（channel, height, width）= (3, 33, 33)
    conv1_param:１層目のフィルタパラメータ
    conv2_param:２層目のフィルタパラメータ
    conv3_param:出力層のフィルタパラメータ
    """
    def __init__(self,  input_dim=(3, 33, 33),
                        conv1_param={'filter_num':64, 'filter_size':9, 'padding':0, 'stride':1, 'learning_rate':1e-4},
                        conv2_param={'filter_num':32, 'filter_size':1, 'padding':0, 'stride':1, 'learning_rate':1e-4},
                        conv3_param={'filter_num':32, 'filter_size':5, 'padding':0, 'stride':1, 'learning_rate':1e-5},
                        weight_init_std=1e-3):
        # フィルター情報のコピー
        filter_num1 = conv1_param['filter_num']
        
        
        # パラメータの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(conv1_param['filter_num']
                                                            , input_dim[0]              # 3
                                                            , conv1_param['filter_size']
                                                            , conv1_param['filter_size'])
        self.params['b1'] = np.zeros(conv1_param['filter_num'])
        self.params['W2'] = weight_init_std * np.random.randn(conv2_param['filter_num']
                                                            , conv1_param['filter_num'] # 64
                                                            , conv2_param['filter_size']
                                                            , conv2_param['filter_size'])
        self.params['b2'] = np.zeros(conv2_param['filter_num'])
        self.params['W3'] = weight_init_std * np.random.randn(input_dim[0]
                                                            , conv2_param['filter_num'] # 32
                                                            , conv3_param['filter_size']
                                                            , conv3_param['filter_size'])
        self.params['b2'] = np.zeros(input_dim[0])
        
        # 各層の生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.parmas['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.parmas['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Conv3'] = Convolution(self.parmas['W3'], self.params['b3'])
        
        self.loss = None
        self.y = None
        self.t = None
        
        
    # 推論(順伝播)
    def predict(self, x):
        # 入力 x に対して各層で推論した結果で x を更新していく
        for layer in self.layers.values():
            x = layer.forward(x)
            
        # 最終的に推論した結果を返却
        return x
        
    # 損失演算
    def loss(self, x, t):
        # 入力 x に対する推論結果 y から損失演算を行う
        y = self.predict(x)
        
        # 最終出力結果と正解データの最小二乗誤差を算出
        # 逆伝播時の最初の入力値となる
        return mean_squared_error(y, t)
        
    # 勾配計算
    def gradient(self, x, t):
        # 順伝播
        loss = self.loss()
        
        # 逆伝播
        # 最後の層の逆伝播から実行するため層の順番を反転させる
        layers = list(self.layers.values())
        layers.reverse()
        dout = 1
        for layer in layers.values():
            layer.backward(dout)
        
        
        
        return grads

#im = cv2.imread('Train/t1.bmp', cv2.IMREAD_GRAYSCALE)
