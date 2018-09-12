import tensorflow as tf
import numpy as np

import IOUtil
from RaniForecastModel import RainForecastModel

time_step = 61
stop_in_step = 31
learningRate = 0.001

batch_size = 20
width = 100
height = 100

out_path = '/home/wingsby/SRAD.tf'

def main():

    with tf.variable_scope('model', reuse=None) as training_scope:
        exampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
        model = RainForecastModel(exampleBatch, prefix='train')

    with tf.variable_scope('val_model', reuse=None):
        valExampleBatch = IOUtil.readBatchData(out_path, batch_size, time_step, width, height)
        val_model = RainForecastModel(valExampleBatch, prefix='val')

    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)
    # Make training session.
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess)

    for itr in range(100000):
        # Generate new batch of data.
        feed_dict = {model.inputs: exampleBatch}
        cost, _ = sess.run([model.loss, model.train_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % 100 == 2:
            # Run through validation set.
            feed_dict = {val_model.inputs: valExampleBatch}
            val_loss, val_summary_str = sess.run([val_model.train_op, val_model.loss],
                                          feed_dict)
        if (itr) % 5000 == 2:
            tf.logging.info('Saving model.')
            saver.save(sess, '/dpdata/rain.saver'+str(itr))


    tf.logging.info('Saving model.')
    saver.save(sess, '/dpdata/rain.saver')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    main()


