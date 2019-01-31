import os

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.interpolate import Rbf

import IOUtil

import matplotlib.pyplot as plt
import imageio

from labs.prediction_train import predict

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

out_path = '/home/wingsby/test'
num_iterations = 10000
event_log_dir = '/home/wingsby/test'
sequence_length = 13
index = 7

batch_size = 2
learning_rate = 0.001
width, height = 64, 64
owidth, oheight = 501, 501
data_path = '/data/SRAD21.tf'
out_path = '/data/output4'


def meter_interpolate(z):
    x, y = np.mgrid[0:owidth, 0:owidth]
    ox, oy = np.mgrid[0:width, 0:width] * owidth / width
    ox = ox * owidth / width
    oy = oy * owidth / width

    # tck = interpolate.bisplrep(np.array(lng).astype(float), np.array(lat).astype(float),np.array(z).astype(float), s=0)
    # znew = interpolate.bisplev(lngnew[:, 0], latnew[0, :], tck)
    # znew = griddata((np.array(lng).astype(float), np.array(lat).astype(float)),np.array(z).astype(float),
    #                 (lngnew, latnew), method='nearest')
    func = Rbf(np.array(ox).astype(float), np.array(oy).astype(float),
               np.array(z).astype(float), function='linear', smooth=0.01)
    znew = func(x, y)
    return znew


def resizeImg(data):
    wimg = Image.fromarray(data * 255)
    wimg = wimg.resize([oheight, owidth])
    wimg = np.round(np.tile(np.reshape(wimg, [oheight, owidth, 1]), [1, 1, 3])).astype(np.uint8)
    return wimg


def Img(data):
    wimg = Image.fromarray(data * 255)
    wimg = np.round(np.tile(np.reshape(wimg, [height, width, 1]), [1, 1, 3])).astype(np.uint8)
    return wimg


model_path = '/home/wingsby/test/model20'  # 恢复网络结构
init = tf.global_variables_initializer()
with tf.variable_scope('model', reuse=None):
    # images = IOUtil.readSingleData(data_path,  sequence_length, width, height)
    images = IOUtil.readBatchData([data_path], batch_size, sequence_length, width, height)
    output, loss, lr = predict(images, sequence_length,
                               prefix='train')
# print('a')
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    # sess.run(init)
    saver.restore(sess, model_path)
    # graph = tf.get_default_graph()
    # out=graph.get_collection('pred_network')[0]
    # input = graph.get_tensor_by_name('images')
    # images = IOUtil.readBatchData(data_path, batch_size, sequence_length, width, height)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for i in range(200):
        images_data = sess.run(images)
        # fig = plt.figure()
        # plt.imshow(images_data[0, 30, :, :] * 255)
        # fig.show()
        feed_dict = {images: images_data,
                     lr: learning_rate}
        # cost, _ = sess.run([loss, train_op],  #
        #                         feed_dict=feed_dict)
        gen = sess.run(output, feed_dict=feed_dict)
        for k in range(batch_size):
            if not os.path.exists(out_path + '/%d' % (i * batch_size + k)):
                os.mkdir(out_path + '/%d' % (i * batch_size + k))
            for j in range(int((sequence_length - 1) / 2)):
                try:
                    oriname = out_path + '/%d/%d.png' % (i * batch_size + k, j)
                    forename = out_path + '/%d/fore%d.png' % (i * batch_size + k, j)
                    ori = resizeImg(images_data[k, 7 + j, :, :])
                    fore = resizeImg(np.reshape(gen[j][k, :, :], [width, width]))
                    # fore = Img(np.reshape(gen[j][k, :, :], [width, height]))
                    plt.imsave(oriname, ori)
                    # plt.imsave(forename, gen[j][k, :, :])
                    plt.imsave(forename, fore)
                except:
                    print(oriname)
