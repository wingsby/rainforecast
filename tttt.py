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

# a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
#
# a1 = tf.tile(a, [1, 2,1])
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(a1))
# a=np.array([[2,3,4],[2,3,4]])
# b=np.array([[5,3,4],[5,3,4]])
# aa=tf.Variable(a,dtype=tf.int16)
# bb=tf.Variable(b,dtype=tf.int16)
#
# d=tf.equal(aa,2)
# e=tf.less(bb,4)
# # cc=tf.Variable(0,dtype=tf.float32)
# dd,_=tf.metrics.true_positives(d,e,name="test")
# df,_=tf.metrics.true_negatives(d,e,name="test")
# dg,_=tf.metrics.false_positives(d,e,name="test")
#
# sz=tf.size(dd,out_type=tf.float32)
# f=dd*2+(df*dg)/sz
# # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test")
# # running_vars_initializer = tf.variables_initializer(var_list=running_vars)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # sess.run(running_vars_initializer)
#     sess.run(tf.local_variables_initializer())
#     print(sess.run(e))
#     print(sess.run(d))
#     # print(sess.run(cc))
#     print(sess.run(f))
# a=np.array([[2,3,4],[2,3,4]])
# b=np.array([[5,3,4],[5,3,4]])
def fun1():
    a=np.array([[2,3,4],[2,3,4]])
    b=np.array([[5,3,4],[5,3,4]])
    true = tf.Variable(a, dtype=tf.float32)
    pred = tf.Variable(b, dtype=tf.float32)
    btt = tf.less(true, 3)
    btp = tf.less(pred, 3)
    sz = tf.size(btt, out_type=tf.float32)
    _, neg_cor = tf.metrics.false_negatives(btt, btp)
    _, hits = tf.metrics.true_positives(btt, btp)
    _, fa_alm = tf.metrics.false_positives(btt, btp)
    _, miss = tf.metrics.true_negatives(btt, btp)

    expCor = ((hits + miss + 0.0) * (hits + fa_alm) + (neg_cor + miss + 0.0) * (neg_cor + fa_alm)) / sz
    hss = (hits + neg_cor - expCor + 0.0) / (sz - expCor)
    return expCor,hss

expCor,hss=fun1()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # print(sess.run(btt))
    # print(sess.run(btp))
    for i in range(500):
        _ec, _hss = sess.run([expCor,hss])

        print((_ec))
        print(_hss)

    # _sz, _hits, _neg_cor, _fa_alm, _miss, _ec, _hss = sess.run([sz, hits, neg_cor, fa_alm, miss, expCor, hss])
    #
    # print("%d %d %d %d %d %d " % (_sz, _hits, _neg_cor, _fa_alm, _miss, _ec))
    # print(_hss)
