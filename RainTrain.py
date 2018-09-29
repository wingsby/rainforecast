import tensorflow as tf
import numpy as np

import IOUtil
from RaniForecastModel import RainForecastModel
import matplotlib.pyplot as plt

time_step = 13
stop_in_step = 7
learningRate = 0.001

batch_size = 20
width = 50
height = 50

out_path = '/home/wingsby/SRAD.tf'


def main():
    with tf.variable_scope('model', reuse=None) as training_scope:
        exampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
        model = RainForecastModel(exampleBatch, prefix='train')

    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    # Make training session.
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess)
    for itr in range(10000):
        # Generate new batch of data.
        # feed_dict = {model.images: exampleBatch}
        # imgs = sess.run(model.images)
        # plt.imshow(imgs[0,0,:,:])
        # plt.show()
        cost, tnvll, hits, neg, fa, miss, sz,sample,sample_t= sess.run(
            [model.loss, model.train_op, model.hits, model.neg_cor, model.fa_alm, model.miss, model.sz,model.sample,model.sample_t])
        # btt = tf.less(btt[0,:,:,0], 100)
        # btp = tf.less(btp[0,:,:,0], 100)
        # hits1, _ = tf.metrics.true_positives(btt, btp)
        # neg_cor1,_=tf.metrics.false_negatives(btt,btp)
        # fa_alm1,_=tf.metrics.false_positives(btt,btp)
        # miss1,_=tf.metrics.true_negatives(btt,btp)
        # sz=tf.size(btt,out_type=tf.float32)
        # expCor = ((hits + miss) * (hits + fa_alm1) +(neg_cor1 + miss) * (neg_cor1 + fa_alm1))
        # hss1 = (hits + neg_cor1 - expCor) / (sz - expCor)
        # sess.run(hss1)
        # print(hss1)
        expCor = ((hits + miss) * (hits + fa) + (neg + miss) * (neg + fa))/(sz+sz*itr)
        hss = (hits + neg - expCor) / (sz+sz*itr - expCor)
        if (itr)%2000==0:
            plt.imshow(sample[0,:,:,0])
            plt.show()
            plt.imshow(sample_t[0,:,:,0])
            plt.show()
            print(str(cost))
        if (itr) % 50 == 2:
            print(str(itr) + ' ' + str(cost) + ' hits: ' + str(hits) + 'neg_cor: ' + str(neg) + 'fa_alm: ' + str(
                fa) + 'miss: ' + str(miss) + 'HSS: ' + str(hss)+' expec:'+str(expCor)+"sz: "+str(sz))
        # Print info: iteration #, cost.::
        tf.logging.info(str(itr) + ' ' + str(cost))
        # if (itr) % 100 == 2:
        #     # Run through validation set.
        #     feed_dict = {val_model.images: valExampleBatch}
        #     val_loss, val_summary_str = sess.run([val_model.train_op, val_model.loss]feed_dict)
        if (itr) % 1000 == 999:
            tf.logging.info('Saving model.')
            saver.save(sess, '/dpdata/rain.saver' + str(itr))
    tf.logging.info('Saving model.')
    saver.save(sess, '/dpdata/rain.saver')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    main()
