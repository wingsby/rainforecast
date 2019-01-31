import os
import random
import re

import keras
from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.layers import Input, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, LSTM, InputLayer, \
    UpSampling3D, Lambda, Conv2DTranspose, Conv3DTranspose
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_exp.multigpu.optimizers import RMSPropMGPU
from matplotlib import pyplot as plt

from scipy.misc import imread

import keras_exp.multigpu

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_size = 6
nb_epoch = 500
# forecast_path='/dpdata/SRAD2018_Test_2/'
# data_path = '/home/wingsby/data/white/'
data_path='/data/SRAD/'
model_path = '/dpdata/res100'
val_path='/dpdata/testval1/'
steps_per_epoch = 2000

height = 100
width = 100
oheight, owidth = 501, 501
frames = 6

from keras_exp.multigpu import (
    get_available_gpus, print_mgpu_modelsummary, ModelMGPU)


def image_read(file_path):
    image = imread(file_path)
    return image


def generate_arrays_from_file(data_path, batch_size):
    X = []
    Y = []
    cnt = 0
    while 1:
        dirs = os.listdir(data_path)
        random.shuffle(dirs)
        for dir in dirs:  # 遍历filenames读取图片
            filenames = os.listdir(data_path + dir)
            # if(random.randint(0,9)>2):
            #    continue
            try:
                filenames.sort(key=lambda x: int(x[-7:-4]))
            except Exception:
                print(dir)
                continue
            except ValueError:
                print(dir)
                continue
            fcnt = 0
            seq_x = []
            seq_y = []
            seq_tmp=[]
            yt=0
            try:
                for filename in filenames:
                    if re.match('.+?png', filename):
                        if (fcnt > 30 and fcnt <= 60 and fcnt % 5== 0):
                            yt+=1
                            image = image_read(data_path + dir + '/' + filename)
                            ind=np.where(image<=80)
                            ind2=np.where(image>100)
                            image[ind]=image[ind]*3+15
                            image[ind2]=0
                            image = Image.fromarray(image)
                            image = image.resize([height, width])
                            image = np.array(image)[:, :, 0:1] / 255.0
                            seq_tmp.append(image.copy().astype(np.float32))
                        elif (fcnt > 0 and fcnt <= 30 and fcnt % 5 == 0):
                            image = image_read(data_path + dir + '/' + filename)
                            image = Image.fromarray(image)
                            image = image.resize([height, width])
                            image = np.array(image)[:, :, 0:1] / 255.0
                            seq_x.append(image.astype(np.float32))
                            ind = np.where(image <= 80)
                            ind2 = np.where(image > 100)
                            image[ind] = image[ind] * 3 + 15
                            image[ind2] = 0
                            # print(filename)
                            # print(fcnt)
                        fcnt += 1
                if (seq_x.__len__() == frames and seq_tmp.__len__() == frames):
                    cnt += 1
                    # 添加seq_y的同一大小数据
                    for i in range(frames-1):
                        seq_x.append(np.zeros(shape=[height, width, 1]).astype(np.float32))
                        seq_y.append(seq_x[i+1])
                    for j in range(frames):
                        seq_y.append(seq_tmp[j])
                    X.append(np.array(seq_x))
                    Y.append(np.array(seq_y))
                if (cnt % batch_size == 0 and X.__len__() == batch_size):
                    yield (np.array(X), np.array(Y))
                    X = []
                    Y = []
            except OSError as e:
                print(e)


def lossfun(y_pred, y_true):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)) * (1 + y_true))


#
def getEncoderForecatModel(ngpus, batch_size):
    model = Sequential()
    #     encoder 3layers  downsampling
    # model.add(Conv3D(filters=32,kernel_size=(1,3,3),input_shape=(None, width, height, 1),batch_size=batch_size,
    #                  padding='same',strides=(1,5,5)))
    # model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7), input_shape=(None, width, height, 1),
                         padding='same', return_sequences=True,strides=(5,5),
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3, parallel_gpu=ngpus, batch_size=batch_size))
    model.add(BatchNormalization())

    # model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
    #                      activation='tanh', recurrent_activation='hard_sigmoid',
    #                      kernel_initializer='glorot_uniform', unit_forget_bias=True,
    #                      dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    # model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

   # model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
    #                      activation='tanh', recurrent_activation='hard_sigmoid',
     #                     kernel_initializer='glorot_uniform', unit_forget_bias=True, strides=(2, 2),
      #                    dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
   # model.add(BatchNormalization())

    #     forecast 3layers upsamping
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

    #model.add(Conv3DTranspose(filters=32, kernel_size=(1,3,3), padding='same', strides=(1, 2, 2)))
    #model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

    model.add(Conv3DTranspose(filters=32, kernel_size=(1,3,3), padding='same', strides=(1, 5, 5)))

    model.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))

    model.add(Lambda(lambda x: x[:, -1 * frames:, :,:,:]))
    return model


def getModel(ngpus, batch_size):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7), input_shape=(None, width, height, 1),
                         padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3, parallel_gpu=ngpus, batch_size=batch_size))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, parallel_gpu=ngpus))
    model.add(BatchNormalization())

    # model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
    #                      activation='tanh', recurrent_activation='hard_sigmoid',
    #                      kernel_initializer='glorot_uniform', unit_forget_bias=True,
    #                      dropout=0.4, recurrent_dropout=0.3,parallel_gpu=ngpus))
    # model.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))
    #model.add(Lambda(lambda x: x[:, -1 * frames:, :,:,:]))

    # model.add

    return model


class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(model_path + ('_%d.h5' % epoch))


# model = getModel()
# single
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')
# cbk = MyCbk(model)
# history=model.fit_generator(generate_arrays_from_file(data_path,batch_size=batch_size),
#                     steps_per_epoch=steps_per_epoch,nb_epoch=nb_epoch,max_q_size=1000,verbose=1,
#                      callbacks=[cbk])


# two
gpus_list = get_available_gpus()
ngpus = len(gpus_list)
model = getModel(ngpus, batch_size)
#model = getEncoderForecatModel(ngpus, batch_size)
# model = buildModel(ngpus)
# print_mgpu_modelsummary(model)
# model_rpn = ModelMGPU(serial_model=model, gdev_list=gpus_list)
model_rpn = multi_gpu_model(model, ngpus)
model_rpn.summary()
optimizer = Adam(lr=0.0012, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True)
model_rpn.compile(loss='mean_squared_error', optimizer=optimizer)
# model_rpn.compile(loss='mean_squared_error', optimizer='Adam')

cbk = MyCbk(model_rpn)
history = model_rpn.fit_generator(generate_arrays_from_file(data_path, batch_size=batch_size),
                                  steps_per_epoch=steps_per_epoch, nb_epoch=nb_epoch, callbacks=[cbk], verbose=1)
                                  # , validation_data=generate_arrays_from_file(data_path, batch_size=500), validation_steps=100)
