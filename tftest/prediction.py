import numpy as np
import tensorflow as tf

import IOUtil
from tftest.prediction_train import Model
import PIL as pil
import matplotlib.pyplot as plt


out_path='/home/wingsby/test'
num_iterations = 10000
event_log_dir = '/home/wingsby/test'
sequence_length = 61
index = 30

batch_size = 8
learning_rate = 0.001
width, height = 40, 40
data_path = '/home/wingsby/SRAD.tf'


model_path='/home/wingsby/test/model92' #恢复网络结构
saver = tf.train.import_meta_graph(model_path + '.meta')
with tf.Session() as sess:

       saver.restore(sess, model_path)
       graph = sess.graph
       # input = graph.get_tensor_by_name('images')
       # images = IOUtil.readBatchData(data_path, batch_size, sequence_length, width, height)
       # graph.get_collection("variables","model")
       images = IOUtil.readBatchData(out_path, batch_size, sequence_length, width, height)
       model = Model(images, sequence_length,
                     prefix='train')
       sess = tf.InteractiveSession()
       for i in range(20):

          # feed_dict = {model.iter_num: np.float32(i),
          #           model.lr: learning_rate}
          # cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
          #                              feed_dict)
          gen=sess.run([model.output])
          fig=plt.figure()
          fig.imshow(gen[0,0,:,:,:]*255)
          fig.show()

