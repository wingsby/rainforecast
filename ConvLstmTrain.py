import os

import tensorflow as tf
import numpy as np

# 建立前馈网络
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell

from CELL import ConvLSTMCell
import matplotlib.pyplot as plt
import IOUtil

hidden_units1 = 50
hidden_units2 = 50
time_step = 13
stop_in_step = 7
hidden_layer = 2
learningRate = 0.0005

batch_size =15
width = 40
height = 40

out_path = '/home/wingsby/SRAD.tf'


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# def forword(inputdata):
#
#     finputdata = tf.tile(tf.expand_dims(inputdata, -1), [1, 1, 1, 1, hidden_units1])
#     cell1 = ConvLSTMCell(shape=[width, height], filters=hidden_units1, kernel=[5, 5])
#     cell2 = ConvLSTMCell(shape=[width,height],filters=hidden_units2,kernel=[3,3])
#     multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2])
#     # multi_rnn_cell.zero_state(batch_size,dtype=tf.uint8)
#     # cell1=BasicLSTMCell()
#     # tf.nn.static_rnn
#     outputs, final_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs=finputdata, dtype=tf.float32)
#     weights = tf.truncated_normal([time_step, 3, 3, hidden_units1, 1], stddev=0.1)
#     conv1 = tf.constant(0.1)
#     out = tf.nn.relu(tf.nn.conv3d(outputs, weights, padding='SAME', strides=[1, 1, 1, 1, 1])) + conv1
#     # tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
#     return out, final_state


def forword(inputdata):
    finputdata = tf.tile(tf.expand_dims(inputdata, -1), [1, 1, 1, 1, hidden_units1])
    cell1 = ConvLSTMCell(shape=[width, height], filters=hidden_units1, kernel=[5, 5])
    cell2 = ConvLSTMCell(shape=[width, height], filters=hidden_units2, kernel=[3, 3])
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

    outputs, final_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs=finputdata, dtype=tf.float32)
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


# def HSSLoss(true, pred):
#     """L2 distance between tensors true and pred.
#
#     Args:
#       true: the ground truth image.
#       pred: the predicted image.
#     Returns:
#       mean squared error between ground truth and predicted image.
#     """
#     #  hits/false alarm/correct neg/miss
#     hits, neg_cor, fa_alm, miss, sz = 0, 0, 0, 0, 0
#     for step in range(stop_in_step - 1, time_step - 1):
#         ctrue = true[:, step + 1, :, :].copy()
#         cpred = pred[:, step, :, :].copy()
#         ctrue[ctrue < 255] = 1
#         cpred[ctrue < 255] = 1
#         right = ctrue[ctrue == cpred]
#         hits += tf.size(right[right == 1])
#         neg_cor += tf.size(right[right > 200])
#         wrong = ctrue[ctrue != cpred]
#         fa_alm += tf.size(wrong[wrong > 200])
#         miss += tf.size(wrong[wrong == 1])
#         sz += tf.size(ctrue)
#     expCor = ((hits + miss) * (hits + fa_alm) +
#                        (neg_cor + miss) * (neg_cor + fa_alm))/sz
#     hss = (tf.cast((hits + neg_cor),tf.float64) - expCor) / (tf.cast((hits + neg_cor),tf.float64) - expCor)
#     return 1 - hss


def HSSLoss(true, pred):
    """L2 distance between tensors true and pred.

        Args:
          true: the ground truth image.
          pred: the predicted image.
        Returns:
          mean squared error between ground truth and predicted image.
        """
    # hits, neg_cor, fa_alm, miss, sz = 0, 0, 0, 0, 0
    ttrue=tf.slice(true,[0,stop_in_step - 1,0,0],[batch_size,time_step-stop_in_step,width,height])
    tpred=tf.slice(pred,[0,stop_in_step,0,0],[batch_size,time_step-stop_in_step,width,height])
    # ttrue=tf.slice(true,[0,stop_in_step - 1,0,0],[batch_size,time_step-1,width,height])
    # tpred=tf.slice(pred,[0,stop_in_step,0,0],[batch_size,time_step,width,height])
    # 生成boolean类型

    # btt=tf.cast(tf.less(ttrue,255),tf.float16)
    # btp=tf.cast(tf.less(tpred,255),tf.float16)
    # return tf.nn.softmax_cross_entropy_with_logits(labels=ttrue,logits=tpred)
    # hits,_=tf.metrics.true_positives(btt,btp)
    # neg_cor,_=tf.metrics.false_negatives(btt,btp)
    # fa_alm,_=tf.metrics.false_positives(btt,btp)
    # miss,_=tf.metrics.true_negatives(btt,btp)
    # sz=tf.size(btt,out_type=tf.float32)
    # expCor = ((hits + miss) * (hits + fa_alm) +(neg_cor + miss) * (neg_cor + fa_alm))
    # hss = (hits + neg_cor - expCor) / (sz - expCor)
    # return 1 - hss
    return tf.reduce_sum(tf.square((ttrue - tpred))) / tf.to_float(tf.size(tpred))


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
    loss = HSSLoss(y_target,outputs)
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train_step = optimizer.minimize(tf.reduce_sum(loss))
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver(max_to_keep=4)
        for i in range(0, 100):
            train_step.run()
            temp_loss, out = sess.run([loss, outputs])
            if (i + 1) % 50 == 0:
                print('Step #' + str(i + 1))
                # temp_loss,out = sess.run([loss,outputs])
                plt.imshow(out[0, 0, :, :])
                plt.show()
                print('Loss = ' + str(temp_loss))
                # loss_batch.append(temp_loss)
                # saver.save(sess, "/dpdata/rain", global_step=i)
        saver.save(sess, "/dpdata/rain")


def forecast():

    exampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
    x_data = exampleBatch
    y_target = exampleBatch
    outputs, final_state = forword(x_data)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        # module_file = tf.train.latest_checkpoint('/dpdata/checkpoint')
        saver.restore(sess, '/dpdata/rain')
        # 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        # prev_seq = train_x[-1]
        predict = []
        # 得到之后100个预测结果
        for i in range(100):
            out=sess.run([outputs])
        # 以折线图表示结果
        plt.figure()
        # plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        # plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()

if __name__ == "__main__":
    train()
    forecast()

