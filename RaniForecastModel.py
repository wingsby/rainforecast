import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.python.framework import dtypes

from CELL import ConvLSTMCell
from ConvLSTMCell import basic_conv_lstm_cell

time_step = 13
stop_in_step = 7
learningRate = 0.001

batch_size = 10
width = 50
height = 50

hidden_units1 = 64
kenel1 = [3, 3]
kenel1_size = 3

hidden_units2 = 64
kenel2 = [3, 3]
kenel2_size = 3

hidden_units3 = 64
kenel3 = [1]
kenel3_size = 1


class RainForecastModel(object):

    # images 输入 [batch_size,time_size,width,height]
    def __init__(self, images, prefix='train', scope=None):
        self.prefix = prefix
        self.images = images
        # 处理images
        postImages = []
        # tmp = tf.placeholder([batch_size, width, height, hidden_units1], dtype=dtypes.float32)
        for i in range(time_step):
            tmp = tf.reshape(tf.slice(images, [0, i, 0, 0], [batch_size, 1, width, height]),
                             [batch_size, width, height])
            tmp = tf.tile(tf.expand_dims(tmp, -1), [1, 1, 1, hidden_units1])
            postImages.append(tmp)
        gen_images = self.build_model(postImages)
        # self.__train__(gen_images, postImages)
        self.loss, self.train_op = self.__train__(gen_images, postImages)
        self.hits, self.neg_cor, self.fa_alm, self.miss, self.hss, self.btt, self.btp = self.__HSS__(gen_images[3],
                                                                                                     tf.slice(
                                                                                                         postImages[
                                                                                                             3 + stop_in_step],
                                                                                                         [0, 0, 0, 0],
                                                                                                         [batch_size,
                                                                                                          width, height,
                                                                                                          1]))
        # return gen_images

    #  images tuple of image
    def build_model(self, images):
        #         decode/train(in)
        #  hidden_units 不等
        # cstate = tf.truncated_normal(shape=[batch_size,  width, height,hidden_units1], stddev=0.1)
        # hstate = tf.truncated_normal(shape=[batch_size,  width, height,hidden_units1], stddev=0.1)
        # state=tf.tuple((cstate,hstate))
        state = tf.truncated_normal(shape=[batch_size, width, height, hidden_units1 * 2], stddev=0.1)
        with slim.arg_scope(
                [basic_conv_lstm_cell, slim.layers.conv2d],
                reuse=tf.AUTO_REUSE):
            for i in range(0, stop_in_step):
                output, state = self.__Cell__(images[i], state, flag=False)
            #         forecast/encode
            gen_images = []
            for i in range(stop_in_step, time_step):
                output, state = self.__Cell__(output, state, flag=True)
                gen_images.append(output)
                output = tf.tile(output, [1, 1, 1, hidden_units1])
            return gen_images

    def __Cell__(self, inputs, state, flag=None):
        # hidden1, lstm_state1 = basic_conv_lstm_cell(shape=[width, height], filters=hidden_units1, kernel=kenel1,scope='h1',reuse=True).call(inputs,
        #                                                                                                       state)
        # hidden2, lstm_state2 = basic_conv_lstm_cell(shape=[width, height], filters=hidden_units2, kernel=kenel1,scope='h2',reuse=True).call(hidden1,
        #                                                                                                       lstm_state1)
        # hidden3, lstm_state3 = basic_conv_lstm_cell(shape=[width, height], filters=hidden_units3, kernel=kenel1,scope='h3',reuse=True).call(hidden2,
        #                                                                                                                  lstm_state2)
        hidden1, lstm_state1 = basic_conv_lstm_cell(inputs, state, num_channels=hidden_units1, filter_size=kenel1_size,
                                                    scope='h1', reuse=tf.AUTO_REUSE)
        hidden2, lstm_state2 = basic_conv_lstm_cell(hidden1, lstm_state1, num_channels=hidden_units2,
                                                    filter_size=kenel2_size, scope='h2', reuse=tf.AUTO_REUSE)
        hidden3, lstm_state3 = basic_conv_lstm_cell(hidden2, lstm_state2, num_channels=hidden_units3,
                                                    filter_size=kenel3_size, scope='h3', reuse=tf.AUTO_REUSE)
        if (flag):
            weights = tf.truncated_normal([3, 3, hidden_units3, 1], stddev=0.1)
            conv1 = tf.constant(0.1)
            out = tf.nn.relu(tf.nn.conv2d(hidden3, weights, padding='SAME', strides=[1, 1, 1, 1])) + conv1
            return out, lstm_state3
        else:
            return hidden3, lstm_state3

    def __loss__(self, pred, true):
        return tf.reduce_sum(tf.square((true - pred) / 256.0)) / tf.to_float(tf.size(pred)) * 10000

    def __HSS__(self, pred, true):
        # ttrue = tf.slice(true, [0, stop_in_step - 1, 0, 0], [batch_size, time_step - stop_in_step, width, height])
        # tpred = tf.slice(pred, [0, stop_in_step, 0, 0], [batch_size, time_step - stop_in_step, width, height])
        # ttrue=tf.slice(true,[0,stop_in_step - 1,0,0],[batch_size,time_step-1,width,height])
        # tpred=tf.slice(pred,[0,stop_in_step,0,0],[batch_size,time_step,width,height])
        # 生成boolean类型
        btt = tf.less(true, 100)
        btp = tf.less(pred, 100)

        _, neg_cor = tf.metrics.false_negatives(btt, btp)
        _, hits = tf.metrics.true_positives(btt, btp)
        _, fa_alm = tf.metrics.false_positives(btt, btp)
        _, miss = tf.metrics.true_negatives(btt, btp)
        sz = tf.size(btt, out_type=tf.float32)
        expCor = ((hits + miss) * (hits + fa_alm) + (neg_cor + miss) * (neg_cor + fa_alm))
        hss = (hits + neg_cor - expCor) / (sz - expCor)
        # print(str(hits)+" "+str(neg_cor)+" "+str(miss)+" "+str(fa_alm)+" "+str(hss))
        return hits, neg_cor, fa_alm, miss, hss, true, pred

    def __train__(self, gen_images, inputs):
        loss = 0.0
        for i in range(0, time_step - stop_in_step):
            loss += self.__loss__(gen_images[i],
                                  tf.slice(inputs[i + stop_in_step], [0, 0, 0, 0], [batch_size, width, height, 1]))
        # train_op = tf.train.AdamOptimizer(learningRate).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
        return loss, train_op
