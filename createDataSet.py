# 文件夹

import os

import re

import numpy as np
from PIL import Image

from scipy.misc import imread

data_path = '/data/lwhite/'
out_path = '/home/wingsby/data/white/'
# data_path = '/data/black/'
# out_path = '/home/wingsby/data/black/'

height = 501
width = 501


def image_read(file_path):
    image = imread(file_path)
    return image


if not os.path.exists(out_path):
    os.makedirs(out_path, mode=0o777)
dirs = os.listdir(data_path)
cnt = 0
for dir in dirs:  # 遍历filenames读取图片
    filenames = os.listdir(data_path + dir)
    cnt += 1
    if (cnt % 1000 == 0):
        print(dir)
    if not os.path.exists(out_path + dir):
        os.mkdir(out_path + dir)
    try:
        filenames.sort(key=lambda x: int(x[-7:-4]))  #
        try:
            for filename in filenames:
                if re.match('.+?png', filename):
                    image = image_read(data_path + dir + '/' + filename)
                    idx = np.where(image <= 80)
                    image[:] = 0
                    image[idx] = image[idx] * 3 + 15
                    image = Image.fromarray(image)
                    # image = image.resize([height, width])
                    wfile = out_path + dir + '/' + filename
                    image.save(wfile)
        except OSError as e:
            print(e)
            print(filename)
            continue
    except Exception as e2:
        print(e2)
        print(dir)
    except ValueError as e3:
        print(e3)
        print(dir)
