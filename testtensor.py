
import tensorflow as tf
import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print (a)
b=tf.constant(a)
print(b)
with tf.Session() as sess:
    t=b.eval()
    for x in b.eval():      #b.eval()就得到tensor的数组形式
        print (x)

    print ('a是数组',a)

    tensor_a=tf.convert_to_tensor(a)
    print ('现在转换为tensor了...',tensor_a)