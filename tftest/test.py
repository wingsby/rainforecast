

import tensorflow as tf;
import imghdr
import os
import time
import re


# imgType = imghdr.what('/home/wingsby/test.jpg')
data_path = '/data/SRAD/'
out_path = '/data/SRAD3.tf'
# writer = tf.python_io.TFRecordWriter(out_path)
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
            if not re.match('.+\.png',filename):
                continue
            # img_raw = tf.gfile.FastGFile(data_path + dir + '/' + filename, 'rb').read()
            type=imghdr.what(data_path + dir + '/' + filename)
            if not type:
                print(data_path + dir + '/' + filename)
            tmp = re.findall('_\d{3}\.png', filename)
        #     exampleImgs[tmp[0][1:4]] = img_raw
        #     # print(img)# 'label': tf.train.Feature(bytes=tf.train.BytesList(value=[bytes(sub, encoding = "utf8")])),
        #     # features=tf.trai
        #     features[tmp[0][1:4]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        # realfeatures = tf.train.Features(feature=features)
        # example = tf.train.Example(features=realfeatures)
        # writer.write(example.SerializeToString())  # 序列化为字符串
        cnt += 1
    except:
        print(dir)
    #if (cnt > 20000):

        #break
    if (cnt % 1000 == 999):
        print("iter:" + str(cnt))
        print("耗时:" + str(time.time() - stime) + "s")
        stime = time.time()

# writer.close()