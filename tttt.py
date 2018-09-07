import tensorflow as tf
import numpy as np
# 创建输入数据
X = np.random.randn(2, 10, 8)

# # 第二个example长度为6
# X[1,6:] = 0
# X_lengths = [10, 6]
#
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
# outputs, last_states = tf.nn.dynamic_rnn(
#     cell=cell,
#     dtype=tf.float64,
#     sequence_length=X_lengths,
#     inputs=X)
# print(outputs)
# for t in outputs:
#     print(t)

a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

a1 = tf.tile(a, [1, 2,1])
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(a1))