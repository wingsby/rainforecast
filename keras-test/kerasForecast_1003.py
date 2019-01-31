import os
import re

import keras
import numpy as np
from PIL import Image
from scipy.misc import imread
from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf
import keras.layers as KL

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# model_path = '/dpdata/black_0.h5'
model_path = '/dpdata/keras100_1003_3.h5'
forecast_path = '/dpdata/Forecast1004/'
# data_path = '/dpdata/SRAD2018_lwhite/'
# data_path = '/dpdata/SRAD2018_mwhite/'
# data_path = '/dpdata/SRAD2018_black/'
data_path = '/dpdata/SRAD2018_Test_2/'

height = 100
width = 100
oheight, owidth = 501, 501
batch_size=7
y_std = 50.0


def image_read(file_path):
    image = imread(file_path)
    return image



model = keras.models.load_model(model_path)
if not os.path.exists(forecast_path):
    os.mkdir(forecast_path )
for root, dirs, filenames in os.walk(data_path):
    for sub in dirs:  # 遍历filenames读取图片
        X=[]
        for subroot, dirs, subfilenames in os.walk(root + sub):
            fcnt = 0
            seq_x = []
            for filename in subfilenames:
                # nPos = filename.index('.png')
                # 前30个 train/后30个 loss
                if re.match('.+?png', filename):
                    try:
                        image = image_read(subroot + '/' + filename)
                        # image = np.reshape(image, [1, oheight, owidth, 3])
                        image = Image.fromarray(image)
                        image = image.resize([height, width])
                        image = np.array(image)[:, :, 0:1] / 255
                        # image = tf.image.resize_bicubic(image[:,:,:,0:0], )
                        if (fcnt > 0 and fcnt<=30 and fcnt % 5 == 0):
                            seq_x.append(image)
                        fcnt += 1
                    except OSError as e:
                        print(e)
            out = model.predict(np.reshape(np.array(seq_x), [1, seq_x.__len__(), height, width, 1]))
            # print(out)
            idx = [0,1,2,3,4,5]
            #if (seq_x.__len__() < 15):
            #    idx = [1, 4, 6, 9, 11, 13]
            if not os.path.exists(forecast_path + sub):
                os.mkdir(forecast_path + sub)
            for i in range(6):
                tmp=out[0, idx[i], :, :, 0]
                # 判断规则???
                # tmp[np.where(np.where(tmp>200))]=255
                wimg = Image.fromarray(out[0, idx[i], :, :, 0] * 255)
                wimg = wimg.resize([oheight, owidth])
                wimg = np.round(np.tile(np.reshape(wimg, [oheight, owidth, 1]), [1, 1, 3])).astype(np.uint8)
                wfile = (forecast_path + sub + '/' + sub + '_' + 'f00%d.png') % (i+1)
                plt.imsave(wfile, wimg)



def fn_keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred * y_std) - (y_true * y_std))))

