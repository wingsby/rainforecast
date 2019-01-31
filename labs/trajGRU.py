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

"""Convolutional LSTM implementation."""
import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers


def init_state(inputs,
               state_shape,
               state_initializer=tf.zeros_initializer(),
               dtype=tf.float32):
    """Helper function to create an initial state given inputs.

    Args:
      inputs: input Tensor, at least 2D, the first dimension being batch_size
      state_shape: the shape of the state.
      state_initializer: Initializer(shape, dtype) for state Tensor.
      dtype: Optional dtype, needed when inputs is None.
    Returns:
       A tensors representing the initial state.
    """
    if inputs is not None:
        # Handle both the dynamic shape as well as the inferred shape.
        inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
        dtype = inputs.dtype
    else:
        inferred_batch_size = 0
    initial_state = state_initializer(
        [inferred_batch_size] + state_shape, dtype=dtype)
    return initial_state


H = 10
W = 10
L = 17


def warp(h, u, v, i, j):
    res = 0
    for m in range(H):
        for n in range(W):
            res += h * tf.maximum(0., 1.0 - tf.abs(i + v - m)) * tf.maximum(0., 1.0 - tf.abs(j + u - n))
    return res


@add_arg_scope
def TrajGRUCell(inputs,
                state,
                num_channels,
                filter_size=5,
                forget_bias=1.0,
                scope=None,
                reuse=None):
    """Basic LSTM recurrent network cell, with 2D convolution connctions.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    Args:
      inputs: input Tensor, 4D, batch x height x width x channels.
      state: state Tensor, 4D, batch x height x width x channels.
      num_channels: the number of output channels in the layer.
      filter_size: the shape of the each convolution filter.
      forget_bias: the initial value of the forget biases.
      scope: Optional scope for variable_scope.
      reuse: whether or not the layer and the variables should be reused.

    Returns:
       a tuple of tensors representing output and the new state.
    """
    spatial_size = inputs.get_shape()[1:3]
    ti = tf.range(spatial_size[0])
    tj = tf.range(spatial_size[1])
    # chn = inputs.get_shape()[3]
    mi = tf.meshgrid(ti, tj)[0]
    mj = tf.transpose(mi)
    mi = tf.cast(tf.reshape(mi, [1, spatial_size[0], spatial_size[1], 1]), dtype=tf.float32)
    mj = tf.cast(tf.reshape(mj, [1, spatial_size[1], spatial_size[0], 1]), dtype=tf.float32)
    if state is None:
        state = init_state(inputs, list(spatial_size) + [num_channels])
    with tf.variable_scope(scope,
                           'TrajGRUCell',
                           [inputs, state],
                           reuse=reuse):
        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        # ow, h = tf.split(axis=3, num_or_size_splits=2, value=state)
        h = state
        # h=state
        inputs_h = tf.concat(axis=3, values=[inputs, h])
        # TrajGRU
        u = layers.conv2d(inputs_h, num_channels, [5, 5], scope="u")
        v = layers.conv2d(inputs_h, num_channels, [5, 5], scope="v")

        z_r_o = layers.conv2d(inputs,
                              3 * num_channels, [filter_size, filter_size],
                              stride=1,
                              activation_fn=None,
                              scope='Gates')

        z, r, o = tf.split(axis=3, num_or_size_splits=3, value=z_r_o)
        # ============== 开写
        mi = tf.tile(mi, [inputs.get_shape()[0], 1, 1, num_channels])
        mj = tf.tile(mj, [inputs.get_shape()[0], 1, 1, num_channels])
        wp = warp(h, u, v, mi, mj)
        # if not warpfields:
        #     warpfields = []
        # warpfields.append(wp)
        # sumw = 0
        # for i in range(L):
        #     tscope = "w%d" % i
        #     if i < warpfields.__len__():
        #         tmp = warpfields[-1 * i]
        #         w = layers.conv2d(tmp, 3 * num_channels, [1, 1], stride=1,
        #                           scope=tscope)
        #         sumw += w
        #     else:
        #         tmp = warpfields[-1]
        #         w = layers.conv2d(tmp, 3 * num_channels, [1, 1], stride=1,
        #                           scope=tscope)
        w = layers.conv2d(wp, 3 * L, [1, 1], stride=1,scope='warp')
        zw, rw, ow = tf.split(axis=3, num_or_size_splits=3, value=w)
        # w = tf.reduce_sum(w, axis=3)
        # sumw += w
        # w = tf.reduce_sum(w, axis=0)
        zw = tf.reshape(tf.reduce_sum(zw, axis=3),
                        [inputs.get_shape()[0], inputs.get_shape()[1], inputs.get_shape()[2], 1])
        rw = tf.reshape(tf.reduce_sum(rw, axis=3),
                        [inputs.get_shape()[0], inputs.get_shape()[1], inputs.get_shape()[2], 1])
        ow = tf.reshape(tf.reduce_sum(ow, axis=3),
                        [inputs.get_shape()[0], inputs.get_shape()[1], inputs.get_shape()[2], 1])

        z = tf.sigmoid(z + tf.tile(zw, [1, 1, 1, num_channels]))
        r = tf.sigmoid(r + tf.tile(rw, [1, 1, 1, num_channels]))
        # 论文中f函数,作者没描述,这里用tanh替代
        # hp = tf.tanh(o + tf.multiply(r, sumw))
        hp = tf.tanh(o + r * tf.tile(ow, [1, 1, 1, num_channels]))
        # new_h = tf.multiply(1 - z, hp) + tf.multiply(z, h)
        new_h = (1 - z) * hp + z * h
        return new_h, new_h
