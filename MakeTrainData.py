#!/usr/bin/env python3
# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm

def MakeLowResolutionImage(org_path='./Train_HR/', redu_path='./Train_LR/', redu_rate=2):
    # オリジナルの訓練データの読み込み
    org_img_list = glob.glob(org_path + '*')
    
    # オリジナルの訓練データは 91 枚
    org_img_num = len(org_img_list)
    
    for img_file in org_img_list:
        img = cv2.imread(img_file)   # BGR 形式
        org_y, org_x, org_c = img.shape # 縦 x 横 x チャンネルであることに注意
        
        # 縮小後のサイズ
        redu_x, redu_y = int(org_x/redu_rate), int(org_y/redu_rate)
        
        # 縮小
        img = cv2.resize(img, (redu_x, redu_y)) # ここは x, yの順番
        
        # ガウシアンフィルタの適用
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 再拡大
        img = cv2.resize(img, (org_x, org_y))
        
        # 低解像画像のファイル名生成
        fname = os.path.basename(img_file)
        ftitle, fext = os.path.splitext(fname)
        output_fpath = redu_path + ftitle + '_LR' + fext   # ./Train_HR\t1.bmp → ./Train_LR\t1_LR.bmp
        
        # 低解像画像の保存
        cv2.imwrite(output_fpath ,img)
        
        break


def SplitImage(input_path='./Train_LR/', output_path='./Train_LR_sub/', fsub_size=33):
    # オリジナルの訓練データの読み込み
    input_img_list = glob.glob(input_path + '*')
    
    # オリジナルの訓練データは 91 枚
    input_img_num = len(input_img_list)
    
    pbar = tqdm(input_img_list)
    for img_file in pbar:
        pbar.set_description("Processing {}".format(input_path))
        # ex)img_file = ./Train_LR\t10_LR.bmp
        img = cv2.imread(img_file)   # BGR 形式
        h, w, _ = img.shape # 縦 x 横 x チャンネルであることに注意
        
        # 均等に分割できないと np.split() が使えないので画像をクロップする
        # 分割数の算出
        [dev_h, dev_w] = np.floor_divide([h, w], fsub_size)
        
        # 余剰画素を算出する
        surplus_h = h % fsub_size
        surplus_w = w % fsub_size
        
        # クロップのオフセット位置を算出する
        # 幅
        lcrop = surplus_w // 2
        rcrop = surplus_w - lcrop
        wcrop = w - surplus_w
        # 高さ
        tcrop = surplus_h // 2
        bcrop = surplus_h - tcrop
        hcrop = h - surplus_h
        
        # クロップ画像を生成する
        #crop_img = img[tcrop:-bcrop, lcrop:-rcrop]
        # ↑はダメ。
        # ex) surplus_h = 0 の場合、tcrop = 0, bcrop = 0 となり、
        #     img[0:-0, *:*]となるためデータが空になる
        crop_img = img[tcrop:(tcrop+hcrop), lcrop:(lcrop+wcrop)]
        
        # np.split()のための分割数
        hcrop_num = hcrop // fsub_size
        wcrop_num = wcrop // fsub_size
        
        out_img = []
        # クロップ画像を分割する
        for h_img in np.vsplit(crop_img, hcrop_num):    # 高さ方向
            for w_img in np.hsplit(h_img, wcrop_num):   # 幅方向
                out_img.append(w_img)
        out_img = np.array(out_img)
        #print(out_img.shape)   # 分割数 x 画像高さ x 画像幅 x チャンネル
        
        """
        # 分割画像の確認
        fig, ax_list = plt.subplots(hcrop_num, wcrop_num, figsize=(5, 5))
        for sub_img, ax in zip(out_img, ax_list.ravel()):
            ax.imshow(sub_img[..., ::-1])
            ax.set_axis_off()
        plt.show()
        """
        
        # 分割した画像のファイル名生成
        fname = os.path.basename(img_file)  # ex)./Train_LR\t10_LR.bmp → t10_LR.bmp
        ftitle, fext = os.path.splitext(fname)  # ex)"t10_LR", ".bmp"
        #print('{},{},{}'.format(fname, ftitle, fext))
        
        
        for idx in range(out_img.shape[0]):
            output_fpath = output_path + ftitle + '_' + str(idx) + fext   # ./Train_HR\t1.bmp → ./Train_LR\t1_LR.bmp
            #print(output_fpath)
            
            # 画像の保存
            cv2.imwrite(output_fpath ,out_img[idx])
            

if __name__ == '__main__':
    MakeLowResolutionImage()
    # 低解像画像の分割
    SplitImage(input_path='./Train_LR/', output_path='./Train_LR_split/', fsub_size=33)
    
    # 正解画像の分割
    SplitImage(input_path='./Train_HR/', output_path='./Train_HR_split/', fsub_size=33)
    
    
