# coding:utf-8
# @Author: wangye
# @Description:
# @Date:Created  on 17:39 2018/9/1.
# @Modify
#======================shaun=======================================
import os

import re
from PIL import Image
# import tensorflow as tf

data_path='/dpdata/SRAD2018_Test_2/'
out_path='/dpdata/SRAD2018_FORECAST2/'

for root, dirs, filenames in os.walk(data_path):
    for sub in dirs:  # 遍历filenames读取图片
        for subroot, dirs, subfilenames in os.walk(root+sub):
            for filename in subfilenames:
                if re.match('.+030\.png',filename) :
                    img = open(subroot + '/' + filename,'rb')
                    img_raw=img.read()
                    if not os.path.exists(out_path+sub):
                        os.mkdir(out_path+sub)
                    for  i in range(1,7):
                        wfile=(out_path+sub+'/'+sub+'_'+'f00%d.png')%i
                        wfileid=open(wfile,'wb')
                        wfileid.write(img_raw)
                        wfileid.close()
                    img.close()
