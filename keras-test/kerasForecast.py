import os
import re

import keras
import numpy as np
from PIL import Image
from scipy.misc import imread
from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# model_path = '/dpdata/black_0.h5'
model_path = '/dpdata/kerasfinal_9.h5'
forecast_path = '/dpdata/Forecast1008/'
# data_path = '/dpdata/SRAD2018_lwhite/'
# data_path = '/dpdata/SRAD2018_mwhite/'
# data_path = '/dpdata/SRAD2018_black/'
data_path = '/dpdata/SRAD2018_Test_2/'

height = 50
width = 50
oheight, owidth = 501, 501
batch_size = 20
y_std = 50.0


def image_read(file_path):
    image = imread(file_path)
    return image


model = keras.models.load_model(model_path)
if not os.path.exists(forecast_path):
    os.mkdir(forecast_path)
dirs = os.listdir(data_path)
cnt = 0
for dir in dirs:
    filenames = os.listdir(data_path + dir)
    seq_x = []
    try:
        filenames.sort(key=lambda x: int(x[-7:-4]))  #
    except Exception:
        print(dir)
    except ValueError:
        print(dir)
    fcnt = 0
    try:
        for filename in filenames:
            if re.match('.+?png', filename):
                if (fcnt > 16 and fcnt % 2 == 0):
                    image = image_read(data_path + dir + '/' + filename)
                    image = Image.fromarray(image)
                    image = image.resize([height, width])
                    image = np.array(image)[:, :, 0:1] / 255.0
                    seq_x.append(image)
                fcnt += 1
        cnt += 1
    except OSError as e:
        print(e)
        continue

    out = model.predict(np.reshape(np.array(seq_x), [1, seq_x.__len__(), height, width, 1]))
    out2 = model.predict(out)
    # print(out)
    idx = [1, 2, 3, 1, 2, 3]
    # if (seq_x.__len__() < 15):
    #     idx = [1, 4, 6, 9, 11, 13]
    if not os.path.exists(forecast_path + dir):
        os.mkdir(forecast_path + dir)
    for i in range(6):
        if i < 3:
            tmp = out[0, idx[i], :, :, 0]
        else:
            tmp = out2[0, idx[i], :, :, 0]
        # 判断规则???
        # tmp[np.where(np.where(tmp>200))]=255
        wimg = Image.fromarray(out[0, idx[i], :, :, 0] * 255)
        wimg = wimg.resize([oheight, owidth])
        wimg = np.round(np.tile(np.reshape(wimg, [oheight, owidth, 1]), [1, 1, 3])).astype(np.uint8)
        wfile = (forecast_path + dir + '/' + dir + '_' + 'f00%d.png') % (i + 1)
        plt.imsave(wfile, wimg)


def fn_keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred * y_std) - (y_true * y_std))))
