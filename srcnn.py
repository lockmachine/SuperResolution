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
        # (64, 3, 9, 9)
        self.params['W1'] = weight_init_std * np.random.randn(conv1_param['filter_num']
                                                            , input_dim[0]              # 3
                                                            , conv1_param['filter_size']
                                                            , conv1_param['filter_size'])
        # (64,)
        self.params['b1'] = np.zeros(conv1_param['filter_num'])
        # (32, 64, 1, 1)
        self.params['W2'] = weight_init_std * np.random.randn(conv2_param['filter_num']
                                                            , conv1_param['filter_num'] # 64
                                                            , conv2_param['filter_size']
                                                            , conv2_param['filter_size'])
        # (32,)
        self.params['b2'] = np.zeros(conv2_param['filter_num'])
        # (3, 32, 5, 5)
        self.params['W3'] = weight_init_std * np.random.randn(input_dim[0]
                                                            , conv2_param['filter_num'] # 32
                                                            , conv3_param['filter_size']
                                                            , conv3_param['filter_size'])
        # (3,)
        self.params['b3'] = np.zeros(input_dim[0])
        
        """
        # for Debug
        for key in self.params.keys():
            print(str(key))
            print(self.params[key][0])
        """
        
        # 各層の生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'])
        
        self.loss_value = None
        self.y = None
        self.t = None
        self.i = 0
        
        
    # 推論(順伝播)
    def predict(self, x):
        # 入力 x に対して各層で推論した結果で x を更新していく
        #print('forward')
        # x = (100, 3, 33, 33)
        for layer in self.layers.values():
            x = layer.forward(x)
            #print(str(layer))
            #print(x[0,0,0:2])
            
        # 最終的に推論した結果を返却
        return x
        
    # 損失演算
    def loss(self, x, t):
        # 入力 x に対する推論結果 y から損失演算を行う
        y = self.predict(x)
        #print('y=')
        #print(y[0, 0, 0])
        #print(t[0, 0, 0])
        #if self.i >= 3:
        #    exit()
        self.i += 1
        # 最終出力結果と正解データの最小二乗誤差を算出
        # 逆伝播時の最初の入力値となる
        #mse = 0.5 * np.sum((y-t)**2)
        batch_size, C, M, N = y.shape
        self.loss_value = mean_squared_error(y, t) / (batch_size*C*M*N)
        #print(self.loss_value)
        return self.loss_value
        
    # 勾配計算
    def gradient(self, x, t):
        # 順伝播
        #loss = self.loss(x, t)
        y = self.predict(x)
        
        # 逆伝播
        # 最後の層の逆伝播から実行するため層の順番を反転させる
        layers = list(self.layers.values())
        layers.reverse()
        batch_size = x.shape[0] # 100
        height = x.shape[2] # 33
        width = x.shape[3] # 33
        
        dout = (y - t) / (batch_size * height * width)
        
        #print('')
        #print('backward')
        for layer in layers:
            dout = layer.backward(dout)
            #print(str(layer) + '=')
            #print(dout[0,0,0])
        #print('')
        
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3'] = self.layers['Conv3'].dW
        grads['b3'] = self.layers['Conv3'].db
        
        return grads
        
    def accuracy(self, x, t):
        #y = self.predict(x)
        sqrtMSE = np.sqrt(self.loss_value)
        acc = 20*np.log10(255/sqrtMSE)
        print(acc)
        return acc
        
    """
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, value in self.params.items():
            params[key] = value
        with open(file_name, "wb") as f:
            pickle.dump(params, f)
            
    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, value in params.items():
            self.params[key] = value
        
        # 各層のパラメーターに戻す
        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i+1)]
            self.layers[key].b = self.params["b" + str(i+1)]
            
    """

if __name__ == '__main__':
    im = cv2.imread('Train_HR/t1.bmp') # BGR形式
    cv2.imshow('t1.bmp',im[:,:,0])  # B
    cv2.waitKey(0)
    cv2.imshow('t1.bmp',im[:,:,1])  # G
    cv2.waitKey(0)
    cv2.imshow('t1.bmp',im[:,:,2])  # R
    print(im.shape) # ex) 176 x 197 x 3
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #im = cv2.imread('Train/t1.bmp', cv2.IMREAD_GRAYSCALE)
