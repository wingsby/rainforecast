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

batch_size=14
parallel_gpu=2
nb_epoch=20
forecast_path='/dpdata/SRAD2018_Test_2/'
data_path='/data/SRAD/'
model_path='/dpdata/keras50_changedata'
steps_per_epoch=20000

height=50
width=50
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
                                idx = np.where(image >200)
                                image[idx] = 100
                                # image = np.reshape(image, [1, oheight, owidth, 3])
                                image = Image.fromarray(image)
                                image = image.resize([height, width])
                                image=np.array(image)[:,:,0:1]/100.0
                                # image = tf.image.resize_bicubic(image[:,:,:,0:0], )
                                if(fcnt>30  and fcnt%2==0):
                                    seq_y.append(image)
                                elif(fcnt>0 and fcnt%2==0):
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




def fn_get_model_convLSTM_tframe_1():

    model = Sequential()

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         input_shape=(None, width, height, 1) ,padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                     activation='linear',
                     padding='same', data_format='channels_last'))

    print(model.summary())
    return model


def fn_get_model_convLSTM_tframe_2():

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         input_shape=(None, width, height, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    #    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
    #    model.add(BatchNormalization())
    #    model.add(Dropout(0.2))

    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))

    print(model.summary())

    return model


def fn_get_model_convLSTM_tframe_3():

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                         input_shape=(None, width, height, 1), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding='same', return_sequences=False,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))

    ### !!! try go_backwards=True !!! ###

    print(model.summary())

    return model


def fn_get_model_convLSTM_tframe_4():

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
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

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))

    ### !!! try go_backwards=True !!! ###

    # print(model.summary())

    return model

def fn_get_model_convLSTM_tframe_5():

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                         input_shape=(None, width, height, 1), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=False,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid',
                     padding='same', data_format='channels_last'))

    print(model.summary())

    return model


def fn_run_model(model, X, y, X_val, y_val, batch_size=50, nb_epoch=40 ,verbose=2 ,is_graph=False):
    history = History()
    history = model.fit(X, y, batch_size=batch_size,
                        epochs=nb_epoch ,verbose=verbose, validation_data=(X_val, y_val))
    if is_graph:
        fig, ax1 = plt.subplots(1 ,1)
        ax1.plot(history.history["val_loss"])
        ax1.plot(history.history["loss"])



def fn_keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred*y_std) - (y_true*y_std))))

def fn_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))



#
# from keras_exp.multigpu import (
#     get_available_gpus, make_parallel, print_mgpu_modelsummary, ModelMGPU)

# model = fn_get_model_convLSTM_tframe_1()
# model = fn_get_model_convLSTM_tframe_2()
# model = fn_get_model_convLSTM_tframe_3()
model = fn_get_model_convLSTM_tframe_4()
# model = fn_get_model_convLSTM_tframe_5()

# model_rpn = multi_gpu_model(model, 2)
# model_rpn = make_parallel(model, 2)
# gdev_list = get_available_gpus()
# mgpu_model = make_parallel(serial_model, gdev_list)
# gpus = get_available_gpus()
# ngpus = len(gpus)
# print_mgpu_modelsummary(model)
# model_rpn = ModelMGPU(serial_model=model, gdev_list=gpus)
# model_rpn = make_parallel(model, gpus)
# print_mgpu_modelsummary(model_rpn)
model.summary()

# opt = RMSPropMGPU()
# print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')
# model_rpn.compile(loss='mean_squared_error',optimizer=opt)
# model.compile(loss=fn_rmse, optimizer='adam')
# model.compile(loss=fn_keras_rmse, optimizer='adam')
# model.compile(loss='binary_crossentropy', optimizer='adam') # doesn't reset weights

# x,y=generate_arrays_from_file(data_path,batch_size=batch_size*10)

# history = History()
class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(model_path + ('_%d.h5' % epoch))


# history = History()
cbk = MyCbk(model)


history=model.fit_generator(generate_arrays_from_file(data_path,batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,nb_epoch=nb_epoch,max_q_size=1000,verbose=1,
                     callbacks=[cbk]           )
# traindata=generate_arrays_from_file(data_path,batch_size=batch_size*2).__next__()
# history = model.fit(traindata[0], traindata[1], batch_size=batch_size,
#                         epochs=nb_epoch)
# fig, ax1 = plt.subplots(1 ,1)
# ax1.plot(history.history["val_loss"])
# ax1.plot(history.history["loss"])
# history = History()
# history = model.fit()
# fit model
# fn_run_model(model, X0_train, X1_train, X0_t_val[:120 ,: ,: ,: ,:], X1_t_val[:120 ,:], batch_size=10, nb_epoch=2
#              ,verbose=1 ,is_graph=True)

model.save(model_path)

# -----------------------------------------------------------------------------
# visualise outputs

# many to many
input_frames = 15
output_frames = input_frames
# s_select = 117 # testing on # 3 # 117 # 0 #2 # 1512
#
# # X_input = X0_t_val[s_select ,:input_frames, :, :, :]
# # X_true = X1_t_val[s_select ,:, :, :, :]
#
# generater=generate_arrays_from_file(data_path,batch_size=batch_size)
# X_pred = model.predict(generater.__next__()[0]) # predict
#
#
#
#
# for i in range(0 ,output_frames):
#
#     # create plot
#     fig = plt.figure(figsize=(10, 15))
#
#     # truth
#     ax = fig.add_subplot(122)
#     # ax.text(1, -3, ('true tframe: ' +str(input_frame s + 5 +i)), fontsize=20, color='b')
#     toplot_true = generater.__next__()[1][0,i, :, :,0]
#     # toplot_true[0 ,0] = 0. # ensure same scale as other
#     # toplot_true[0 ,1] = 1.
#     plt.imshow(toplot_true ,cmap='gist_gray_r')
#
#     # predictions
#     ax = fig.add_subplot(121)
#     # ax.text(1, -3, ('predictions tframe: ' +str(input_frame s + 5 +i)), fontsize=20, color='b')
#     toplot_pred = X_pred[0 ,i, :, :, 0]
#     # toplot_pred[0 ,0] = 0. # ensure same scale as other
#     # toplot_pred[0 ,1] = 1.
#     plt.imshow(toplot_pred ,cmap='gist_gray_r')
