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
import sys

sys.path.extend(['/home/wingsby/develop/python/rainforecast'])
# solve cannot import local package
import os

import numpy as np
import IOUtil
import tensorflow as tf


# How often to record tensorboard summaries.
from tftest.PredictionModel import forward

SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
# SAVE_INTERVAL = 2000

# tf record data location:

# local output directory
OUT_DIR = '/home/wingsby/test'
num_iterations = 5000
event_log_dir = '/home/wingsby/test'
sequence_length = 61
index = 30

# batch_size = 8
single_batch_size = 10
learning_rate = 0.0001
width, height = 64, 64
data_path = '/data/SRAD'
val_data_path='/data/SRAD10.tf'
RELU_SHIFT = 1e-12
DNA_KERN_SIZE = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
num_gpus = 2
epoch = 100


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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        try:
            for g, var in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        except:
            print("")
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


def train(images=None,
          sequence_length=None,
          prefix=None,
          reuse_scope=None):
    with tf.device('/cpu:0'):
        tower_grads = []
        summaries = []
        if sequence_length is None:
            sequence_length = sequence_length
        if prefix is None:
            prefix = tf.placeholder(tf.string, [])
        iter_num = tf.placeholder(tf.float32, [])
        # Split into timesteps.
        reuse_vars = False
        for i in range(num_gpus):
            # with tf.device('/gpu:%d' % i):
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                # with tf.name_scope('GPU_%d' % i) as scope:
                single_images = images[i * single_batch_size:(i + 1) * single_batch_size]
                single_images = tf.split(axis=1, num_or_size_splits=int(single_images.get_shape()[1]),
                                         value=single_images)
                single_images = [tf.squeeze(img) for img in single_images]
                # if scope is None:
                gen_images = forward(
                    single_images,
                    30, cdna=True, dna=False, reuse=reuse_vars)
                # output = tf.reshape(gen_images, [31, batch_size, height, width, 1])
                # tf.add_to_collection('pred_network', output)
                output = gen_images
                # else:  # If it's a validation or test model.
                #     with tf.variable_scope(reuse_scope, reuse=True):
                #         gen_images = forward(
                #             single_images,
                #             30, cdna=True, dna=False)
                #         # output = tf.reshape(gen_images, [31, batch_size, height, width, 1])
                #         # tf.add_to_collection('pred_network', output)
                #         output = gen_images
                # L2 loss, PSNR for eval.
                loss, psnr_all = 0.0, 0.0
                for ii, x, gx in zip(
                        range(len(gen_images)), single_images[index:],
                        gen_images):
                    x = tf.reshape(x, [single_batch_size, height, width, 1])
                    recon_cost = mean_squared_error(x, gx)
                    psnr_i = peak_signal_to_noise_ratio(x, gx)
                    psnr_all += psnr_i
                    loss += recon_cost
                    # summaries.append(
                    #     tf.summary.scalar(prefix + '_recon_cost' + str(ii), recon_cost))
                    # summaries.append(tf.summary.scalar(prefix + '_psnr' + str(ii), psnr_i))

                psnr_all = psnr_all
                # loss = loss = loss / np.float32(len(gen_images))
                loss = loss
                lr = tf.placeholder_with_default(learning_rate, ())
                optimizer = tf.train.AdamOptimizer(lr)
                grads = optimizer.compute_gradients(loss)
                tower_grads.append(grads)
                reuse_vars = True
                # tf.summary.scalar(prefix + '_psnr_all', psnr_all)
                # tf.summary.scalar(prefix + '_loss', loss)
                # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        tower_grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(tower_grads)
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar('learning_rate', lr))
        summ_op = tf.summary.merge(summaries)
        return output, train_op, lr, loss, psnr_all, summ_op


def predict(images=None,
            sequence_length=None,
            prefix=None):
    if sequence_length is None:
        sequence_length = sequence_length
    if prefix is None:
        prefix = tf.placeholder(tf.string, [])
    iter_num = tf.placeholder(tf.float32, [])
    # Split into timesteps.
    reuse_vars = False
    # with tf.name_scope('GPU_%d' % i) as scope:
    single_images = images[0: single_batch_size]
    single_images = tf.split(axis=1, num_or_size_splits=int(single_images.get_shape()[1]),
                             value=single_images)
    single_images = [tf.squeeze(img) for img in single_images]
    # if scope is None:
    gen_images = forward(
        single_images,
        30, cdna=True, dna=False, reuse=reuse_vars)
    output = gen_images
    # L2 loss, PSNR for eval.
    loss, psnr_all = 0.0, 0.0
    for ii, x, gx in zip(
            range(len(gen_images)), single_images[index:],
            gen_images):
        x = tf.reshape(x, [single_batch_size, height, width, 1])
        recon_cost = mean_squared_error(x, gx)
        psnr_i = peak_signal_to_noise_ratio(x, gx)
        psnr_all += psnr_i
        loss += recon_cost
    lr = tf.placeholder_with_default(learning_rate, ())
    loss = loss = loss / np.float32(len(gen_images))
    return output, loss, lr


def main():
    print('Constructing models and inputs.')
    with tf.variable_scope('model', reuse=None) as training_scope:
        idxlist=[1,2,3,4,5,6,9,10,11,12,13,14,15,16]
        files=[]
        for i in idxlist:
           files.append(data_path+('%d.tf'%i))
        images = IOUtil.readBatchData(files, single_batch_size * num_gpus, sequence_length, width, height)
        output, train_op, lr, loss, psnr_all, summ_op = train(images, sequence_length,
                                                              prefix='train')

    # with tf.variable_scope('val_model', reuse=None):
    #     val_images = IOUtil.readBatchData(val_data_path, single_batch_size * num_gpus, sequence_length, width, height)
    #     val_output, val_train_op, val_lr, val_loss, val_psnr_all, val_summ_op = train(images, sequence_length,
    #                                                           prefix='val')

    print('Constructing saver.')
    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    # Make training session.
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(
        event_log_dir, graph=sess.graph, flush_secs=10)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess)
    tf.logging.info('iteration number, cost')
    # Run training.
    for epoch_itr in range(epoch):
        for itr in range(num_iterations):
            try:
                # Generate new batch of data.
                if coord.should_stop():
                     print('cord stop')
                     break
                images_data = sess.run(images)
                feed_dict = {images: images_data,
                             lr: learning_rate}
                
                cost, summary_str, _ = sess.run([loss, summ_op, train_op],
                                                feed_dict=feed_dict)
                if (itr) % SUMMARY_INTERVAL==0:
                    summary_writer.add_summary(summary_str, itr)
                    print("第%d轮,%d次"%(epoch_itr,itr))
            except Exception as e:
                #coord.request_stop():
                print('图像发生错误')
                continue

        tf.logging.info('Saving model.')
        saver.save(sess, OUT_DIR + '/model' + str(epoch_itr))

    tf.logging.info('Saving model.')
    saver.save(sess, OUT_DIR + '/model')
    tf.logging.info('Training complete')
    # tf.logging.flush()


if __name__ == '__main__':
    main()
