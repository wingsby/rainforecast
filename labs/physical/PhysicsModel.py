# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""
import time

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python import layers as tf_layers

# Amount to use when lower bounding tensors
from labs.trajGRU import TrajGRUCell
from tftest.lstm_ops import basic_conv_lstm_cell

RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


# images: 一次60张图
# index:从什么时候开始预测
def forward(images, index, reuse=None):
    stime = time.time()
    batch_size, img_height, img_width = images[0].get_shape()[0:3]
    gru_fun = TrajGRUCell
    # Generated robot states and images.
    gen_images = []
    lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    # lstm_state5, lstm_state6, lstm_state7 = None, None, None
    warpfields1, warpfields2, warpfields3, warpfields4 = None, None, None, None
    # warpfields5, warpfields6, warpfields7 = None, None, None
    for i in range(images.__len__()):
        # Reuse variables after the first timestep.
        if i > 0:
            reuse = True
        # reuse = True
        with slim.arg_scope(
                [gru_fun, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):
            if i > index:
                prev_image = tf.reshape(gen_images[-1], [batch_size, img_height, img_width, 1])
            else:
                prev_image = tf.reshape(images[i], [batch_size, img_height, img_width, 1])
            enc0 = slim.layers.conv2d(prev_image, lstm_size[2], 5, stride=2, scope='preinp',normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm1'},)
            hidden1, lstm_state1, warpfields1 = gru_fun(enc0, lstm_state1, warpfields1, lstm_size[0], filter_size=5,
                                                        scope="sate1")
            hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm1')
            hidden2, lstm_state2, warpfields2 = gru_fun(hidden1, lstm_state1, warpfields1, lstm_size[1], filter_size=5,
                                                        scope="sate2")
            hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm2')
            hidden3, lstm_state2, warpfields3 = gru_fun(hidden2, lstm_state2, warpfields2, lstm_size[1], filter_size=5,
                                                        scope="sate3")
            hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm3')
            hidden4, lstm_state4, warpfields4 = gru_fun(hidden3, lstm_state3, warpfields3, lstm_size[1], filter_size=5,
                                                        scope="sate4")
            hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm4')
            enc1 = slim.layers.conv2d_transpose(
                hidden4, hidden4.get_shape()[3], 3, stride=2, scope='convt1')
            output = slim.layers.conv2d(enc1, 1, 1, stride=1, scope='finalout')
            # 最后引入

            if (i > index - 1):
                gen_images.append(output)
        # print(gen_images.name)
    print(time.time() - stime)
    return gen_images


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data points.

    Args:
      ground_truth_x: tensor of ground-truth data points.
      generated_x: tensor of generated data points.
      batch_size: batch size
      num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])
