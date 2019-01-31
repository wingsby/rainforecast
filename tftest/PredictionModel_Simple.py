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
from tftest.lstm_ops import basic_conv_lstm_cell

RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


# images: 一次60张图
# index:从什么时候开始预测
def forward(images, index, dna, cdna, num_masks=10):
    stime=time.time()
    batch_size, img_height, img_width = images[0].get_shape()[0:3]
    lstm_func = basic_conv_lstm_cell

    # Generated robot states and images.
    gen_images = []
    lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
    lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
    lstm_state5, lstm_state6, lstm_state7 = None, None, None

    for i in range(images.__len__()):
        # Reuse variables after the first timestep.
        reuse = (i>0)
        with slim.arg_scope(
                [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):
            if reuse and i > index:
                prev_image = tf.reshape(gen_images[-1], [batch_size, img_height, img_width, 1])
            else:
                prev_image = tf.reshape(images[i], [batch_size, img_height, img_width, 1])

            enc0 = slim.layers.conv2d(
                prev_image,
                32, 5,
                stride=1,
                scope='scale1_conv1',
                normalizer_fn=tf_layers.layer_norm,
                normalizer_params={'scope': 'layer_norm1'})

            hidden1, lstm_state1 = lstm_func(
                enc0, lstm_state1, lstm_size[0], scope='state1')
            hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
            hidden2, lstm_state2 = lstm_func(
                hidden1, lstm_state2, lstm_size[1], scope='state2')
            hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
            enc1 = slim.layers.conv2d(
                hidden2, hidden2.get_shape()[3], [3, 3], stride=1, scope='conv2')

            hidden3, lstm_state3 = lstm_func(
                enc1, lstm_state3, lstm_size[2], scope='state3')
            hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
            hidden4, lstm_state4 = lstm_func(
                hidden3, lstm_state4, lstm_size[3], scope='state4')
            hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm5')
            output = slim.layers.conv2d(
                hidden4, hidden4.get_shape()[3], [3, 3], stride=1, scope='conv3')

            if (i > index - 1):
                gen_images.append(output[:,:,:,0:1])
        # print(gen_images.name)
    print(time.time()-stime)
    return gen_images

