import os
import re
import shutil
import datetime
import time
import logging.handlers

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

owidth = 501
oheight = 501
width = 50
height = 50

fixed_length = 61
logger = None


def initLog():
    # path_log = '/home/app/meteo_server/nwc/log/'
    # 日志：==========
    LOG_FILE = (datetime.datetime.now() - datetime.timedelta(days=0)).strftime("%Y%m%d")
    handler = logging.handlers.RotatingFileHandler(LOG_FILE + '.log',
                                                   maxBytes=1024 * 1024 * 10, backupCount=5,
                                                   encoding='UTF-8')  # 实例化handler
    fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt)  # 实例化formatter
    handler.setFormatter(formatter)  # 为handler添加formatter
    tlogger = logging.getLogger(LOG_FILE)  # 获取名为LOG_FILE的logger
    tlogger.addHandler(handler)  # 为logger添加handler
    tlogger.setLevel(logging.DEBUG)
    return tlogger


# read data and put into array
def writeData():
    data_path = '/data/SRAD/'
    out_path = '/data/SRAD3.tf'
    writer = tf.python_io.TFRecordWriter(out_path)
    cnt = 0
    stime = time.time()
    # outfeature=dict()
    dirs = os.listdir(data_path)
    for dir in dirs:  # 遍历filenames读取图片
        filenames = os.listdir(data_path + dir)
        # try:
        #     filenames.sort(key=lambda x: int(x[-7:-4]))
        # except Exception:
        #     print(dir)
        #     continue
        exampleImgs = dict()
        features = dict()
        try:
            for filename in filenames:
                # img = Image.open(subroot + '/' + filename)
                # img = img.resize((501, 501))
                # img_raw = img.tobytes()  # 将图片转化为二进制格式
                if not re.match('.+\.png', filename):
                    continue
                img_raw = tf.gfile.FastGFile(data_path + dir + '/' + filename, 'rb').read()
                try:
                    im = tf.image.decode_png(data_path + dir + '/' + filename, channels=1)
                except:
                    print(data_path + dir + '/' + filename)
                tmp = re.findall('_\d{3}\.png', filename)
                exampleImgs[tmp[0][1:4]] = img_raw
                # print(img)# 'label': tf.train.Feature(bytes=tf.train.BytesList(value=[bytes(sub, encoding = "utf8")])),
                # features=tf.trai
                features[tmp[0][1:4]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            realfeatures = tf.train.Features(feature=features)
            example = tf.train.Example(features=realfeatures)
            writer.write(example.SerializeToString())  # 序列化为字符串
            cnt += 1
        except:
            print(dir)
        # if (cnt > 20000):

        # break
        if (cnt % 1000 == 999):
            print("iter:" + str(cnt))
            print("耗时:" + str(time.time() - stime) + "s")
            stime = time.time()

    writer.close()


# return outfeature


def readBatchData(filename, batchsize, seq_length, width, height):
    filename_queue = tf.train.string_input_producer(filename)  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    prefeatures = dict()
    for i in range(0, fixed_length):
        prefeatures["%03d" % i] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(serialized_example, features=prefeatures)
    img = []
    skip = int((fixed_length - 1) / (seq_length - 1))
    for i in range(0, seq_length):
        k = i * skip
        # print(k)
        try:
            tmp = tf.image.decode_png(features["%03d" % k], channels=1)
            # tmp = tf.image.resize_image_with_crop_or_pad(tmp, oheight, owidth)
            tmp = tf.reshape(tmp, [1, oheight, owidth, 1])
            tmp = tf.image.resize_bicubic(tmp, [height, width])
            tmp = tf.cast(tmp, tf.float32)
        except:
            print("wrong occurs")
            tmp = tf.zeros(shape=[height, width])
        img.append(tmp)
    img = tf.reshape(img, [seq_length, width, height])
    exampleBatch = tf.train.shuffle_batch([img], batch_size=batchsize, capacity=100, min_after_dequeue=20)
    return exampleBatch



def readSingleData(filename,  seq_length, width, height):
    filename_queue = tf.train.string_input_producer(filename)  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    prefeatures = dict()
    for i in range(0, fixed_length):
        prefeatures["%03d" % i] = tf.FixedLenFeature([], tf.string)
    prefeatures['dir']=tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(serialized_example, features=prefeatures)
    img = []
    skip = int((fixed_length - 1) / (seq_length - 1))
    for i in range(0, seq_length):
        k = i * skip
        # print(k)
        try:
            tmp = tf.image.decode_png(features["%03d" % k], channels=1,dtype=np.uint8)
            # tmp = tf.reshape(tmp, [1, oheight, owidth, 1])
            # tmp = tf.image.resize_bicubic(tmp, [height, width])
            # tmp = tf.cast(tmp, tf.float32)
        except:
            print("wrong occurs")
            # tmp = tf.zeros(shape=[height, width])
        img.append(tmp)
    img = tf.reshape(img, [seq_length, owidth, oheight])
    dir=features['dir']

    # exampleBatch = tf.train.shuffle_batch([img], batch_size=batchsize, capacity=100, min_after_dequeue=20)
    return img,dir


def readDataForForecast(filename, seq_length, width, height):
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
        # tmp = tf.image.resize_image_with_crop_or_pad(tmp, oheight, owidth)
        tmp = tf.reshape(tmp, [1, oheight, owidth, 1])
        tmp = tf.image.resize_bicubic(tmp, [height, width])
        tmp = tf.cast(tmp, tf.float32) / 255
        img.append(tmp)
    img = tf.reshape(img, [seq_length, width, height])
    return img


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
    prefeatures = dict()
    for i in range(0, 61):
        prefeatures["%03d" % i] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(serialized_example, features=prefeatures)
    img = []
    for i in range(0, 61):
        tmp = tf.image.decode_png(features["%03d" % i], channels=1)
        # tmp=tf.reshape(tmp,[501,501])
        img.append(tmp)
    # img = tf.image.decode_png(features['sample_img'], channels=1)
    # img = tf.reshape(img, [501, 501])
    img = tf.reshape(img, [61, 501, 501])
    exampleBatch = tf.train.shuffle_batch([img], batch_size=10, capacity=500, min_after_dequeue=50)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        array = sess.run(exampleBatch)
        array1 = np.reshape(array, [10, 61, 501, 501])
        for i in range(0, 61):
            plt.imshow(array1[0, i, :, :].astype(np.float32))
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


def writePartData(index, size):
    data_path = '/data/SRAD/'
    out_path = '/data/SRAD%d.tf' % index
    writer = tf.python_io.TFRecordWriter(out_path)
    cnt = 0
    stime = time.time()
    # outfeature=dict()
    skipstep = (index - 1) * size
    dirs = os.listdir(data_path)
    dirs.sort()
    for dir in dirs:  # 遍历filenames读取图片
        cnt += 1
        if cnt < skipstep or cnt > skipstep + size:
            continue
        # if (cnt < 250730 or cnt > 250740):
        #     continue
        filenames = os.listdir(data_path + dir)
        # try:
        #     filenames.sort(key=lambda x: int(x[-7:-4]))
        # except Exception:
        #     print(dir)
        #     continue
        exampleImgs = dict()
        features = dict()
        try:
            for filename in filenames:
                # img = Image.open(subroot + '/' + filename)
                # img = img.resize((501, 501))
                # img_raw = img.tobytes()  # 将图片转化为二进制格式
                if not re.match('.+\.png', filename):
                    continue
                img_raw = tf.gfile.FastGFile(data_path + dir + '/' + filename, 'rb').read()
                tmp = re.findall('_\d{3}\.png', filename)
                exampleImgs[tmp[0][1:4]] = img_raw
                # print(img)# 'label': tf.train.Feature(bytes=tf.train.BytesList(value=[bytes(sub, encoding = "utf8")])),
                features[tmp[0][1:4]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                features['dir']=tf.train.Feature(bytes_list=tf.train.BytesList(value=[dir.encode()]))
            realfeatures = tf.train.Features(feature=features)
            example = tf.train.Example(features=realfeatures)
            writer.write(example.SerializeToString())  # 序列化为字符串
        except:
            print(dir)
        # if (cnt > 20000):

        # break
        if (cnt % 1000 == 999):
            print("iter:" + str(cnt))
            print("耗时:" + str(time.time() - stime) + "s")
            stime = time.time()

    writer.close()


def image_read(file_path):
    image = imread(file_path)
    return image


def filterData():
    data_path = '/data/SRAD/'
    bad_path = '/data/bad/'
    dirs = os.listdir(data_path)
    dirs.sort()
    cnt = 0
    for dir in dirs:
        if (cnt % 1000) == 999:
            print("处理到第%d个目录：%s" % (cnt, dir))
            logger.info("处理到第%d个目录：%s" % (cnt, dir))
        cnt += 1

        filenames = os.listdir(data_path + dir)
        if(filenames.__len__()< fixed_length):
             shutil.move(data_path + dir, bad_path)
             continue
        try:
            for filename in filenames:
                if not re.match('.+\.png', filename):
                    shutil.move(data_path + dir + "/" + filename, bad_path)
                    continue
                image = image_read(data_path + dir + '/' + filename)

                if image.shape[1] != 501 and image.shape[2] != 501 and image.shape[3] != 3 and image.size==501*501*3:
                    shutil.move(data_path + dir, bad_path)
                    break
        except:
            print("s%目录错误,发生在第%d个目录中" % (dir, cnt))
            logger.error("s%目录错误,发生在第%d个目录中" % (dir, cnt))
            shutil.move(data_path + dir, bad_path)
            logger.info("完成s%目录移动" % dir)
            continue
    print('处理完成')


if __name__ == "__main__":

    # str='/data/SRAD/RAD_526382404222543/RAD_526382404222543_060.png'
    # im=image_read(str)
    # fig = plt.figure()
    # plt.imshow(im[:, :, 0])
    # fig.show()
    logger = initLog()
    # filterData()
    # writeData()
    # 跑到第6个出错了 13
    for i in range(20,25):
        try:
         writePartData(i + 1, 10000)
        except Exception as e:
            print(e)
        except IOError as e1:
            print(e1)
        except EOFError as e2:
            print(e2)

    # out_path = '/data/SRAD.tf'
    # testRead(out_path)
