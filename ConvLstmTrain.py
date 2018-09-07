import os

import tensorflow as tf;
import numpy as np

# 建立前馈网络
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell

from CELL import ConvLSTMCell

import IOUtil

hidden_units1 = 32
hidden_units2 = 64
time_step = 61
hidden_layer = 2
learningRate = 0.01

batch_size = 20
width = 501
height = 501
out_path = '/home/wingsby/SRAD.tf'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def forword(inputdata):
    # data=[batch_size, max_time, ...]
    # initstates=
    # cell = ConvLSTMCell(shape, filters, kernel)
    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=
    # inputs = tf.placeholder(tf.float32, [batch_size, time_step] + [width,height] + [3])
    finputdata = tf.tile(tf.expand_dims(inputdata, -1), [1, 1, 1, 1, hidden_units1])
    cell1 = ConvLSTMCell(shape=[width,height],filters=hidden_units1,kernel=[7,7])
    # cell2 = ConvLSTMCell(shape=[time_step,width,height],filters=hidden_units2,kernel=[7,7])
    # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2])
    # multi_rnn_cell.zero_state(batch_size,dtype=tf.uint8)
    # cell1=BasicLSTMCell()
    outputs, final_state = tf.nn.dynamic_rnn(cell1, inputs=inputdata, dtype=tf.uint8)

    return outputs, final_state


def train():
    x_data = tf.placeholder(shape=[batch_size,time_step, width, height], dtype=tf.uint8)
    y_target = tf.placeholder(shape=[batch_size,time_step, width, height], dtype=tf.uint8)
    exampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
    init = tf.global_variables_initializer()
    outputs, final_state = forword(x_data)
    # softmax_w = tf.get_variable(
    #     "softmax_w", [size, 2], dtype=tf.uint8)
    # softmax_b = tf.get_variable("softmax_b", [2], dtype=tf.uint8)
    # logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
    # Reshape logits to be a 3-D tensor for sequence loss
    # logits = tf.reshape(outputs, [batch_size, time_step, 2])
    loss = tf.contrib.seq2seq.sequence_loss(
        outputs,
        y_target,
        tf.ones([batch_size, time_step], dtype=tf.int32),
        average_across_timesteps=False,
        average_across_batch=True)
    # 声明优化器
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train_step = optimizer.minimize(tf.reduce_sum(loss))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(0, 1000):
            array = sess.run(exampleBatch)
            array1 = np.reshape(array, [batch_size, time_step, width, height])
            sess.run(train_step, feed_dict={'x_data': exampleBatch, 'y_target': exampleBatch})
        if (i + 1) % 5 == 0:
            # print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: exampleBatch, y_target: exampleBatch})
            print('Loss = ' + str(temp_loss))
            # loss_batch.append(temp_loss)

# def run_epoch(session, model, eval_op=None, verbose=False):
#     """Runs the model on the given data."""
#     start_time = time.time()
#     costs = 0.0
#     iters = 0
#     state = session.run(model.initial_state)
#
#     fetches = {
#         "cost": model.cost,
#         "final_state": model.final_state,
#     }
#     if eval_op is not None:
#         fetches["eval_op"] = eval_op
#
#     for step in range(model.input.epoch_size):
#         feed_dict = {}
#         for i, (c, h) in enumerate(model.initial_state):
#             feed_dict[c] = state[i].c
#             feed_dict[h] = state[i].h
#
#         vals = session.run(fetches, feed_dict)
#         cost = vals["cost"]
#         state = vals["final_state"]
#
#         costs += cost
#         iters += model.input.num_steps
#
#         if verbose and step % (model.input.epoch_size // 10) == 10:
#             print("%.3f perplexity: %.3f speed: %.0f wps" %
#                   (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
#                    iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
#                    (time.time() - start_time)))
#
#     return np.exp(costs / iters)
if __name__ == "__main__":
    train()