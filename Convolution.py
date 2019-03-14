#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.util import im2col, col2im


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # backward 時に使用
        self.x = None
        self.col = None
        self.col_W = None
        
        # 重み・バイアスの勾配
        self.dW = None
        self.db = None
        
    def forward(self, x):
        # フィルターのサイズ取得
        FN, C, FH, FW = self.W.shape
        
        # 入力データのサイズ取得
        N, C, H, W = x.shape
        
        # 出力サイズ(FN, OH, OW)
        OH = int((H + 2*self.pad - FH) / self.stride) + 1
        OW = int((W + 2*self.pad - FW) / self.stride) + 1
        
        col = im2col(x, FH, FW, self.stride, self.pad)  # (OH*OW)x(C*FW*FH)
        col_W = self.W.reshape(FN, -1).T    # フィルターの展開 FNx(FW*FH*C)
        
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
        
    def backward(self, dout):
        # フィルターのサイズを取得
        FN, C, FH, FW = self.W.shape
        
        # 逆伝播の入力データを順伝播のtranspose前の行列形式に変換
        # forward の最終出力が out.reshape.transpose なので
        # backwardの入力を dout.transpose.reshape にする
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        
        """
        順伝播でのバイアス加算は、それぞれのデータ（1個目のデータ、2個目のデータ、・・・）に対して加算が行われる。
        そのため、逆伝播の際には、それぞれのデータの逆伝播の値がバイアスの要素に集約される必要がある。
        """
        # db = dL/dB = dL/dY = dout となるので、1x1xFNの形にする
        self.db = np.sum(dout, axis=0)
        
        # dL/dW = X.T * dL/dY となるので、
        self.dW = np.dot(self.col.T, dout)
        self.dW = (self.dW.T).reshape(FN, C, FH, FW)
        
        # dL/dx = dL/dY * W.T
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad )
        
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None
        
    def forward(self, x):
        N, C, H, W = x.shape
        self.x = x
        OH = int(1 + (H - self.pool_h) / self.stride)
        OW = int(1 + (W - self.pool_w) / self.stride)
        
        # 展開（１）
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # col は (N x OH x OW, C x pool_h x pool_w) になっているので、チャンネル毎に縦に並べる
        # (N x OH x OW x C, pool_h x pool_w) となる
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        # 最大値（２）
        # 各プーリング領域に対して最大値を求める
        # (N x OH x OW x C, 1) のサイズになる
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        
        # 整形（３）
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        
        self.arg_max = arg_max
        self.x = x
        
        return out
        
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
