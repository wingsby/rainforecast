import os
import re

import keras
import numpy as np
from PIL import Image
from scipy.misc import imread
from matplotlib import pyplot as plt
from keras import backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



# model_path = '/dpdata/res100_final_45.h5'
model_path = '/dpdata/res100_8.h5'
forecast_path = '/dpdata/1020/'
# forecast_path = '/dpdata/Forecast1010tmp/'
# forecast_path='/dpdata/val1008/'
# data_path = '/dpdata/SRAD2018_Test_2/'
# data_path='/dpdata/testval1/'
data_path='/dpdata/tmp/'
# data_path='/dpdata/testdata/'
# data_path='/dpdata/processed/'

height = 100
width = 100
oheight, owidth = 501, 501
batch_size=6
frames=6


def image_read(file_path):
    image = imread(file_path)
    return image



# model = keras.models.load_model(model_path)
dict={'frames':6}
model = keras.models.load_model(model_path,custom_objects=dict)
if not os.path.exists(forecast_path):
    os.mkdir(forecast_path)
dirs = os.listdir(data_path)
cnt = 0
X=[]
file=[]
for dir in dirs:
    filenames = os.listdir(data_path + dir)
    seq_x = []
    try:
        print(dir)
        filenames.sort(key=lambda x: int(x[-7:-4]))  #
    except Exception:
        print(dir)
    except ValueError:
        print(dir)
    fcnt = 0
    try:
        for filename in filenames:
            if re.match('.+?png', filename):
                if (fcnt > 0 and fcnt<=30 and fcnt % 5 == 0):
                    image = image_read(data_path + dir + '/' + filename)
                    image = Image.fromarray(image)
                    image = image.resize([height, width])
                    image = np.array(image)[:, :, 0:1] / 255.0
                    seq_x.append(image.astype(np.float32))
                fcnt += 1
        if(seq_x.__len__()==frames):
            for i in range(frames-1):
                seq_x.append(np.ones(shape=[height,width,1]).astype(np.float32))
            X.append(seq_x)
            file.append(dir)
            cnt += 1
        if(X.__len__()==batch_size):
            out = model.predict(np.reshape(np.array(X), [batch_size, 2*frames-1, height, width, 1]))
            for k in range(batch_size):
                seq_x=X[k]
                tsub=file[k]
                if not os.path.exists(forecast_path + tsub):
                    os.mkdir(forecast_path + tsub)
                idx=[0,1,2,3,4,5]
                #idx = [1, 4, 6, 9, 11, 14]
                #if (seq_x.__len__() < 15):
                #    idx = [1, 4, 6, 9, 11, 13]
                if not os.path.exists(forecast_path + tsub):
                    os.mkdir(forecast_path + tsub)
                for i in range(6):
                    tmp = out[k, idx[i], :, :, 0] * 255
                    # ind1=np.where(tmp<2)
                    # ind = np.where(tmp >0)
                    # tmp[ind1]=50
                    # tmp[ind]=(tmp[ind]-10)/3
                    # tmp[ind]=tmp[ind]
                    # 判断规则???
                    wimg = Image.fromarray(tmp)
                    wimg = wimg.resize([oheight, owidth])
                    wimg = np.round(np.tile(np.reshape(wimg, [oheight, owidth, 1]), [1, 1, 3])).astype(np.uint8)
                    wfile = (forecast_path + tsub + '/' + tsub + '_' + 'f00%d.png') % (i + 1)
                    # Image.fromarray(wimg).save(wfile)
                    # plt.imsave(wimg,wfile)
                    plt.imsave(wfile, wimg)
                    plt.imshow(tmp)
                    plt.show()

            file=[]
            X=[]
    except OSError as e:
        print(e)
        continue





