import os
import re

import keras
from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.models import Model
from keras.layers import Input, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, LSTM
from keras import backend as K
from keras.callbacks import History
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras_exp.multigpu.optimizers import RMSPropMGPU
from matplotlib import pyplot as plt

from scipy.misc import imread

import keras_exp.multigpu

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_size=10
nb_epoch=20
# forecast_path='/dpdata/SRAD2018_Test_2/'
# data_path='/data/SRAD/'
data_path='/data/mwhite/'
model_path='/dpdata/keras1005_much'
steps_per_epoch=2000

height=120
width=120
oheight, owidth=501,501

def image_read(file_path):
    image = imread(file_path)
    return image

def generate_arrays_from_file(data_path,batch_size):
    X =[]
    Y =[]
    cnt=0
    while 1:
        for root, dirs, filenames in os.walk(data_path):
            for sub in dirs:  # 遍历filenames读取图片
                for subroot, dirs, subfilenames in os.walk(root + sub):
                    fcnt=0
                    seq_x=[]
                    seq_y=[]
                    try:
                        for filename in subfilenames:
                            # nPos = filename.index('.png')
                            # 前30个 train/后30个 loss
                            if re.match('.+?png', filename):
                                image = image_read(subroot + '/' + filename)
                                idx = np.where(image <= 80)
                                image[idx] = (80-image[idx])*2
                                image = Image.fromarray(image)
                                image = image.resize([height, width])
                                image=np.array(image)[:,:,0:1]/255.0
                                # image = tf.image.resize_bicubic(image[:,:,:,0:0], )
                                if(fcnt>30 and fcnt%5==0):
                                    seq_y.append(image)
                                elif(fcnt>0 and fcnt%5==0):
                                    seq_x.append(image)
                                fcnt += 1

                        cnt+=1
                        X.append(np.array(seq_x))
                        Y.append(np.array(seq_y))
                        if (cnt%batch_size==0):
                            yield (np.array(X), np.array(Y))
                            X=[]
                            Y=[]
                    except OSError as e:
                        print(e)
                        continue


def fn_get_model_convLSTM_tframe_4():

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                         input_shape=(None,width, height, 1), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    #model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
    #                     activation='tanh', recurrent_activation='hard_sigmoid',
    #                     kernel_initializer='glorot_uniform', unit_forget_bias=True,
    #                     dropout=0.4, recurrent_dropout=0.3))
    #model.add(BatchNormalization())


    #model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
    #                     activation='tanh', recurrent_activation='hard_sigmoid',
    #                     kernel_initializer='glorot_uniform', unit_forget_bias=True,
    #                     dropout=0.4, recurrent_dropout=0.3))
    #model.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))

    ### !!! try go_backwards=True !!! ###

    # print(model.summary())

    return model




def fn_keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std))))

def fn_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))



model = fn_get_model_convLSTM_tframe_4()
model.summary()

# opt = RMSPropMGPU()
# print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')

class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(model_path + ('_%d.h5' % epoch))


# history = History()
cbk = MyCbk(model)


history=model.fit_generator(generate_arrays_from_file(data_path,batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,nb_epoch=nb_epoch,max_q_size=1000,verbose=1,
                     callbacks=[cbk])


