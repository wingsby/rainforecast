import os

import numpy as np
import tensorflow as tf

import IOUtil
from IOUtil import readBatchData
from vgg16.vgg import Vgg16

OUT_DIR = '/home/wingsby/test'
num_iterations = 5000
event_log_dir = '/home/wingsby/test'
sequence_length = 61
index = 30

# batch_size = 8
single_batch_size = 10
learning_rate = 0.0001
width, height = 224, 224
data_path = '/data/SRAD'
val_data_path='/data/SRAD10.tf'
RELU_SHIFT = 1e-12
DNA_KERN_SIZE = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
num_gpus = 2
epoch = 100
batch_size=20

idxlist = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
files = []
for i in idxlist:
    files.append(data_path + ('%d.tf' % i))
images=IOUtil.readBatchData(files, batch_size, 61, width, height)
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(
    event_log_dir, graph=sess.graph, flush_secs=10)
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess)
# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:

        images_data = sess.run(images)
        feed_dict = {images: images_data}
        vgg = Vgg16()

        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)
        # utils.print_prob(prob[0], './synset.txt')
        # utils.print_prob(prob[1], './synset.txt')
