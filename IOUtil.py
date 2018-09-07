import os
import re

import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np


# read data and put into array
def writeData():
    data_path = '/dpdata/SRAdata/'
    out_path = '/home/wingsby/SRAD.tf'
    writer = tf.python_io.TFRecordWriter(out_path)
    cnt = 0
    # outfeature=dict()
    for root, dirs, filenames in os.walk(data_path):
        # print(dirs)
        for sub in dirs:  # 遍历filenames读取图片
            for subroot, subdirs, subfilenames in os.walk(root + sub):
                # subfilenames=subfilenames.sort()
                exampleImgs = dict()
                features = dict()
                for filename in subfilenames:
                    # img = Image.open(subroot + '/' + filename)
                    # img = img.resize((501, 501))
                    # img_raw = img.tobytes()  # 将图片转化为二进制格式
                    img_raw=tf.gfile.FastGFile(subroot + '/' + filename, 'rb').read()
                    tmp=re.findall('_\d{3}\.',filename)
                    exampleImgs[tmp[0][1:4]]=img_raw
                    # print(img)# 'label': tf.train.Feature(bytes=tf.train.BytesList(value=[bytes(sub, encoding = "utf8")])),
                    # features=tf.trai
                    features[tmp[0][1:4]]=tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                realfeatures=tf.train.Features(feature=features)
                example = tf.train.Example(features=realfeatures)
                writer.write(example.SerializeToString())  # 序列化为字符串
                # outfeature=features
                cnt += 1
            if (cnt > 1000):
                cnt = 9999
                break

    writer.close()
    # return outfeature


def readBatchData(filename,batchsize,seq_length,width,height):
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    prefeatures = dict()
    for i in range(0, seq_length):
        prefeatures["%03d" % i] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(serialized_example, features=prefeatures)
    img = []
    for i in range(0, seq_length):
        tmp = tf.image.decode_png(features["%03d" % i], channels=1)
        img.append(tmp)
    img = tf.reshape(img, [seq_length, width, height])
    exampleBatch = tf.train.shuffle_batch([img], batch_size=batchsize, capacity=500, min_after_dequeue=50)


def testRead(filename):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    # features = tf.parse_single_example(serialized_example,
    #                                    features={
    #                                        'label': tf.FixedLenFeature([], tf.string),
    #                                        'sample_img': tf.FixedLenFeature([],dtype=tf.string)
                                           # 'sample_img': tf.FixedLenFeature([], tf.string)
                                       # })  # 将image数据和label取出来
    # features=tf.parse_example
    # img = tf.decode_raw(features['sample_img'], tf.uint8)
    # label = tf.decode_raw(features['label'], tf.uint8)
    # array = tf.reshape(img, [501, 501])  # reshape为128*128的3通道图片
    # labStr = tf.reshape(label, [30])
    prefeatures=dict()
    for i in range(0,61):
        prefeatures["%03d" % i]=tf.FixedLenFeature([],tf.string)
    features = tf.parse_single_example(serialized_example,features=prefeatures)
    img=[]
    for i in range(0, 61):
        tmp=tf.image.decode_png(features["%03d" % i], channels=1)
        # tmp=tf.reshape(tmp,[501,501])
        img.append(tmp)
    # img = tf.image.decode_png(features['sample_img'], channels=1)
    # img = tf.reshape(img, [501, 501])
    img=tf.reshape(img,[61,501,501])
    exampleBatch = tf.train.shuffle_batch([img],batch_size=30, capacity=500,min_after_dequeue=50)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        array = sess.run(exampleBatch)
        array1 = np.reshape(array, [30,61,501, 501])
        for i in range(0,61):
           plt.imshow(array1[0,i,:,:].astype(np.float32))
           plt.show()

        # print(str(labStr, encoding='utf-8'))
        # nimg = Image.fromarray(array1, mode="L")  # 这里Image是之前提到的
        # nimg.save('/home/wingsby/test.jpg')  # 存下图片
    return img



# def readData(filename):
#     filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
#
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     prefeatures = dict()
#     for i in range(0, 61):
#         prefeatures["%03d" % i] = tf.FixedLenFeature([], tf.string)
#     features = tf.parse_single_example(serialized_example, features=prefeatures)
#     img = []
#     for i in range(0, 61):
#         tmp = tf.image.decode_png(features["%03d" % i], channels=1)
#         img.append(tmp)
#     array = sess.run(img)
#     array1 = np.reshape(array, [61, 501, 501])
#     plt.imshow(array1[0, :, :].astype(np.float32))
#     plt.show()


from scipy.misc import imread

if __name__ == "__main__":
    # writeData()
    out_path = '/home/wingsby/SRAD.tf'
    testRead(out_path)
