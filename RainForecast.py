import tensorflow as tf
import numpy as np

import IOUtil
from RaniForecastModel import RainForecastModel

time_step = 13
stop_in_step = 7
learningRate = 0.001

batch_size = 5
width = 64
height = 64

out_path = '/home/wingsby/SRAD.tf'

save_model_path='/dpdata/rain.saver'


def main():
    with tf.variable_scope('model', reuse=None) as forecast:
        exampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
        model = RainForecastModel(exampleBatch, prefix='forecast')
    # Make training session.

    # model_file = tf.train.latest_checkpoint('ckpt/')
    # saver.restore(sess, model_file)
    # val_loss, val_acc = sess.run([loss, acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    # print('val_loss:%f, val_acc:%f' % (val_loss, val_acc))
    #
    #
    sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # tf.train.start_queue_runners(sess)
    saver = tf.train.Saver()
    saver.restore(sess, save_model_path)
    graph = tf.get_default_graph()
    for itr in range(100000):
        # Generate new batch of data.
        cost, _ = sess.run([model.loss, model.train_op])
        hss,miss,hits = sess.run([model.hss,model.miss,model.hits])
        if (itr) % 10 == 2:
            print(str(itr) + ' ' + str(cost) + 'hits: ' + str(hits)+ 'miss: ' + str(miss)+ 'HSS: ' + str(hss))
        # Print info: iteration #, cost.
        if (itr) % 5000 == 2:
            tf.logging.info('Saving model.')
            saver.save(sess, '/dpdata/rain.saver' + str(itr))
    tf.logging.info('Saving model.')
    saver.save(sess, '/dpdata/rain.saver')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    main()
