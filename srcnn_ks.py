import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten
from keras.optimizers import Momentum
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def network(input_shape):
    model = Sequential()
    
    # SRCNN ネットワークの構成
    # data_format='channels_last'
    model.add(Conv2D(64, 9, padding='valid', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, 1, padding='valid', activation='relu'))
    model.add(Conv2D(3 , 5, padding='valid', activation=None))
    
    return model
    
class SRDataset():
    def __init__(self):
        self.image_shape = (33, 33, 3)
        
    def get_batch(self):
