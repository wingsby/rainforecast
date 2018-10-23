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

"""Code for training the prediction model."""
import os

import numpy as np

import IOUtil
import tensorflow as tf

# How often to record tensorboard summaries.
from tftest.PredictionModel import construct_model

SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 10

# tf record data location:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# local output directory
OUT_DIR = '/home/wingsby/test'
num_iterations = 100
event_log_dir = '/home/wingsby/test'
sequence_length = 61
index = 30

batch_size = 8
learning_rate = 0.001
width, height = 40, 40
data_path = '/home/wingsby/SRAD.tf'


## Helper functions
def peak_signal_to_noise_ratio(true, pred):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      peak signal to noise ratio (PSNR)
    """
    return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):

    def __init__(self,
                 images=None,
                 sequence_length=None,
                 prefix=None,
                 reuse_scope=None):

        if sequence_length is None:
            sequence_length = sequence_length

        # if prefix is None:
        #     prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        images = tf.split(axis=1, num_or_size_splits=int(images.get_shape()[1]), value=images)
        images = [tf.squeeze(img) for img in images]

        if reuse_scope is None:
            gen_images = construct_model(
                images,
                30, cdna=True, dna=False)
            self.output = gen_images
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images = construct_model(
                    images,
                    30, cdna=True, dna=False)
                self.output = gen_images
        # L2 loss, PSNR for eval.
        loss, psnr_all = 0.0, 0.0
        for i, x, gx in zip(
                range(len(gen_images)), images[index:],
                gen_images):
            x = tf.reshape(x, [batch_size, height, width, 1])
            recon_cost = mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            # summaries.append(
            #     tf.summary.scalar('_recon_cost' + str(i), recon_cost))
            # summaries.append(tf.summary.scalar('_psnr' + str(i), psnr_i))
            loss += recon_cost

        # summaries.append(tf.summary.scalar('_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(gen_images))

        # summaries.append(tf.summary.scalar('_loss', loss))

        self.lr = tf.placeholder_with_default(learning_rate, ())

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        # self.summ_op = tf.summary.merge(summaries)



def main():
    print('Constructing models and inputs.')
    with tf.variable_scope('model', reuse=None) as training_scope:
        images = IOUtil.readBatchData(data_path, batch_size, sequence_length, width, height)
        model = Model(images, sequence_length,
                      prefix='train')

    with tf.variable_scope('val_model', reuse=None):
        val_images = IOUtil.readBatchData(data_path, batch_size, sequence_length, width, height)
        val_model = Model(val_images,
                          sequence_length, prefix='val')

    print('Constructing saver.')
    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)

    # Make training session.
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # summary_writer = tf.summary.FileWriter(
    #     event_log_dir, graph=sess.graph, flush_secs=10)

    # if FLAGS.pretrained_model:
    #   saver.restore(sess, FLAGS.pretrained_model)

    tf.train.start_queue_runners(sess)

    tf.logging.info('iteration number, cost')

    # Run training.
    for itr in range(num_iterations):
        # Generate new batch of data.
        feed_dict = {model.iter_num: np.float32(itr),
                     model.lr: learning_rate}
        # cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
        #                                 feed_dict)
        cost, _ = sess.run([model.loss, model.train_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        tf.logging.info(str(itr) + ' ' + str(cost))

        # if (itr) % VAL_INTERVAL == 2:
        #     # Run through validation set.
        #     feed_dict = {val_model.lr: 0.0,
        #                  val_model.iter_num: np.float32(itr)}
        #     # _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
        #     #                               feed_dict)
        #     _, val_summary_str = sess.run([val_model.train_op],
        #                                   feed_dict)
        #     # summary_writer.add_summary(val_summary_str, itr)

        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model.')
            saver.save(sess, OUT_DIR + '/model' + str(itr))

    tf.logging.info('Saving model.')
    saver.save(sess, OUT_DIR + '/model')
    tf.logging.info('Training complete')
    # tf.logging.flush()


if __name__ == '__main__':
    main()
