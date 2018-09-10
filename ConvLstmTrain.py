import os

import tensorflow as tf
import numpy as np

# 建立前馈网络
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell

from CELL import ConvLSTMCell

import IOUtil

hidden_units1 = 32
# hidden_units2 = 32
time_step = 61
stop_in_step = 31
hidden_layer = 1
learningRate = 0.01

batch_size = 5
width = 100
height = 100

out_path = '/home/wingsby/SRAD.tf'


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True


def forword(inputdata):
    # data=[batch_size, max_time, ...]
    # initstates=
    # cell = ConvLSTMCell(shape, filters, kernel)
    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=
    # inputs = tf.placeholder(tf.float32, [batch_size, time_step] + [width,height] + [3])
    finputdata = tf.tile(tf.expand_dims(inputdata, -1), [1, 1, 1, 1, hidden_units1])
    cell1 = ConvLSTMCell(shape=[width, height], filters=hidden_units1, kernel=[7, 7])
    # cell2 = ConvLSTMCell(shape=[time_step,width,height],filters=hidden_units2,kernel=[7,7])
    # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2])
    # multi_rnn_cell.zero_state(batch_size,dtype=tf.uint8)
    # cell1=BasicLSTMCell()

    outputs, final_state = tf.nn.dynamic_rnn(cell1, inputs=finputdata, dtype=tf.float32)
    weights = tf.truncated_normal([time_step, 3, 3, hidden_units1, 1], stddev=0.1)
    conv1 = tf.constant(0.1)
    out = tf.nn.relu(tf.nn.conv3d(outputs, weights, padding='SAME', strides=[1, 1, 1, 1, 1])) + conv1
    # tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    return out, final_state


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


def HSSLoss(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    #  hits/false alarm/correct neg/miss
    hits, neg_cor, fa_alm, miss, sz = 0, 0, 0, 0, 0
    for step in range(stop_in_step - 1, time_step - 1):
        ctrue = true[:, step + 1, :, :].copy()
        cpred = pred[:, step, :, :].copy()
        ctrue[ctrue < 255] = 1
        cpred[ctrue < 255] = 1
        right = ctrue[ctrue == cpred]
        hits += tf.size(right[right == 1])
        neg_cor += tf.size(right[right > 200])
        wrong = ctrue[ctrue != cpred]
        fa_alm += tf.size(wrong[wrong > 200])
        miss += tf.size(wrong[wrong == 1])
        sz += tf.size(ctrue)
    expCor = ((hits + miss) * (hits + fa_alm) +
                       (neg_cor + miss) * (neg_cor + fa_alm))/sz
    hss = (tf.cast((hits + neg_cor),tf.float64) - expCor) / (tf.cast((hits + neg_cor),tf.float64) - expCor)
    return 1 - hss


def train():
    exampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
    # exampleBatch = tf.cast(exampleBatch, tf.float32)
    x_data = exampleBatch
    y_target = exampleBatch
    outputs, final_state = forword(x_data)
    outputs = tf.reshape(outputs, [batch_size, time_step, width, height])
    # 声明优化器
    # npout = sess.run(outputs)
    # npy = sess.run(y_target)
    loss = HSSLoss(outputs, y_target)
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train_step = optimizer.minimize(tf.reduce_sum(loss))
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        for i in range(0, 1000):
            train_step.run()
            if (i + 1) % 5 == 0:
                print('Step #' + str(i + 1))
                temp_loss = sess.run(loss)
                print('Loss = ' + str(temp_loss))
                # loss_batch.append(temp_loss)


if __name__ == "__main__":
    train()
