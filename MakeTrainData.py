#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

# オリジナルの訓練データの読み込み
org_img_list = glob.glob('./Train/*')

# オリジナルの訓練データは 91 枚
org_img_num = len(org_img_list)

for img_file in range(org_img_list):
    im = cv2.imread(img_file)   # BGR 形式
