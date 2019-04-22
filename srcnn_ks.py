import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
import pickle

def SRCNN_ks(input_shape):
    model = Sequential()
    
    # SRCNN ネットワークの構成
    # data_format='channels_last'
    model.add(Conv2D(64, 9, padding='valid', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, 1, padding='valid', activation='relu'))
    model.add(Conv2D(3 , 5, padding='valid', activation=None))
    
    return model
    
class SRDataset():
    def __init__(self):
        # 学習用画像サイズ
        self.image_shape = (33, 33, 3)
        
    def get_batch(self):
        # 訓練データの読み込み
        with open('./LR_split_data.pkl', 'rb') as f:
            x_train_img = pickle.load(f)
        with open('./HR_split_data.pkl', 'rb') as f:
            t_train_img = pickle.load(f)
            
        x_train_img = x_train_img.astype('float32')
        x_train_img /= 255
        t_train_img = x_train_img.astype('float32')
        t_train_img /= 255
        
        return x_train_img, t_train_img
        

class Trainer():
    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), 'logdir')
        self.model_file_name = 'srcnn_model.hdf5'
        
    def train(self, x_train, t_train, batch_size, epochs, validation_split):
        #if os.path.exists(self.log_dir):
            #import shutil
            #shutil.rmtree(self.log_dir)
        #os.mkdir(self.log_dir)
        
        self._target.fit(
            x_train, t_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(os.path.join(self.log_dir, self.model_file_name), save_best_only=True)
            ],
            verbose=self.verbose
        )

# データセットの読み込み
dataset = SRDataset()

# ネットワークのインスタンス生成
model = SRCNN_ks(dataset.image_shape)

# オプティマイザーのインスタンス生成
optimizer = SGD(lr=0.01, momentum=0.9)

x_train, t_train = dataset.get_batch()
#print(x_train.shape)    # (4364, 33, 33, 3)
#print(t_train.shape)    # (4364, 21, 21, 3)
trainer = Trainer(model, loss='mean_squared_error', optimizer=optimizer)
trainer.train(x_train, t_train, batch_size=128, epochs=12, validation_split=0.0)

#score = model.evaluate()
