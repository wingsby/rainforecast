import tensorflow as tf
import numpy as np

from CELL import ConvLSTMCell

time_step = 61
stop_in_step = 31
learningRate = 0.001

batch_size = 20
width = 100
height = 100

hidden_units1 = 64
kenel1 = [3, 3]

hidden_units2 = 64
kenel2 = [3, 3]

hidden_units3 = 64
kenel3 = [1]


class RainForecastModel(object):


    # images 输入 [batch_size,time_size,width,height]
    def __init__(self,images,prefix='train',scope=None):
        self.prefix=prefix
        self.images=images
        # 处理images
        postImages=[]
        for i in range(time_step):
            tmp=tf.reshape(tf.slice(images, [0, i, 0, 0], [batch_size, 1, width, height]),[batch_size,width,height])
            tmp=tf.tile(tf.expand_dims(tmp, -1), [1, 1, 1, hidden_units1])
            postImages.append(tmp)
        gen_images=self.build_model(postImages)
        self.__train__()
        self.loss,self.train_op=self.__train__(gen_images,postImages)
        return gen_images


    #  images tuple of image
    def build_model(self, images):
        #         decode/train(in)
        #  hidden_units 不等
        cstate = tf.truncated_normal(shape=[batch_size,  width, height,hidden_units1], stddev=0.1)
        hstate = tf.truncated_normal(shape=[batch_size,  width, height,hidden_units1], stddev=0.1)
        state=tf.tuple((cstate,hstate))
        for i in range(0, stop_in_step):
            output, state = self.__Cell__(images[i], state,flag=False)
        #         forecast/encode
        gen_images = []
        for i in range(stop_in_step, time_step):
            output, state = self.__Cell__(output,state,flag=True)
            gen_images.append(output)
        return gen_images

    def __Cell__(self, inputs, state,flag=None):
        hidden1, lstm_state1 = ConvLSTMCell(shape=[width, height], filters=hidden_units1, kernel=kenel1,scope='h1').call(inputs,
                                                                                                              state)
        hidden2, lstm_state2 = ConvLSTMCell(shape=[width, height], filters=hidden_units2, kernel=kenel1,scope='h2').call(hidden1,
                                                                                                              lstm_state1)
        hidden3, lstm_state3 = ConvLSTMCell(shape=[width, height], filters=hidden_units3, kernel=kenel1,scope='h3').call(hidden2,
                                                                                                                         lstm_state2)
        if(flag):
            weights = tf.truncated_normal([3, 3, hidden_units3, 1], stddev=0.1)
            conv1 = tf.constant(0.1)
            out = tf.nn.relu(tf.nn.conv2d(hidden3, weights, padding='SAME', strides=[1, 1, 1, 1])) + conv1
            return out,lstm_state3
        else:
            return hidden3, lstm_state3


    def __loss__(true, pred):
        return tf.reduce_sum(tf.square((true - pred) / 256.0)) / tf.to_float(tf.size(pred)) * 10000

    def __train__(self,gen_images, inputs):
        loss=0.0
        for i in range(0, time_step - stop_in_step):
            loss += self.__loss__(gen_images[i], inputs[i + stop_in_step])
        train_op = tf.train.AdamOptimizer(learningRate).minimize(loss)
        return  loss,train_op
