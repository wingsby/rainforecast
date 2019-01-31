import os
import shutil
import sys
sys.path.append('/home/wingsby/develop/python/rainforecast/')
import IOUtil
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/home/wingsby/test'
num_iterations = 5000
event_log_dir = '/home/wingsby/test'
sequence_length = 61
index = 30

# batch_size = 8
single_batch_size = 1
learning_rate = 0.0001
width, height = 64, 64
data_path = '/data/SRAD/'
val_data_path='/data/SRAD20.tf'
RELU_SHIFT = 1e-12
DNA_KERN_SIZE = 5

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
num_gpus = 1
epoch = 50

SRAD_path = '/data/SRAD'
bad_path = '/data/maybad/'

print('Constructing models and inputs.')
# idxlist=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
idxlist=[24]
files=[]
for i in idxlist:
   files.append(SRAD_path+('%d.tf'%i))
images,dir = IOUtil.readSingleData(files, sequence_length, width, height)

# Make training session.
# sess = tf.InteractiveSession(config=config)
# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()
# tf.train.start_queue_runners(sess)
# for i in range(10000):
#     try:
#         sess.run([images])
#         if(i%100==99):
#             print(i)
#     except:
#         print("error occur at %d" % i)
#
# print("ok")

# Create the graph, etc.
init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph.
sess = tf.Session(config=config)

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# tf.reset_default_graph()
try:
    for i in range(30000):
        # Run training steps or whatever
        try:
            img,cdir=sess.run([images,dir])
            # print(cdir)
        except Exception as e:
            try:
                shutil.move(data_path + str(cdir,'utf-8'), bad_path)
                print(cdir)
            except Exception as e1:
                print(e1)
                print('wrong occurs at %d'% i)

        if (i % 100 == 99):
                print(i)

except Exception as e:
    print(e)
    # print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
try:
  coord.join(threads)
except Exception as e:
    print(e)
    print('do nothing')
print('ok')

sess.close()
