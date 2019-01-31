import tensorflow as tf

width = 10
height = 10

ti = tf.range(width)
tj = tf.range(height)
mi = tf.meshgrid(ti, tj)
mj = tf.meshgrid(ti, tj)
mii=tf.tensordot(ti,tj)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tti, ttj, ttmi, ttmj,tmii = sess.run([ti, tj, mi, mj,mii])
print(tti)
print(ttj)
print(ttmi)
print(ttmj)
print(tmii)
print(tmii)
