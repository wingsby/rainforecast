# 1 直接求误差
import os

import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


def absLoss(foreimg, valimg):
    return np.sum(np.abs(foreimg[:, :, 0] - valimg[:, :, 0]))/np.size(foreimg)


# 2 直接HSS
def directHss(pred, true):
    #  hits/false alarm/correct neg/miss
    hits, neg_cor, fa_alm, miss, sz = 0, 0, 0, 0, 0
    ctrue = true[:, :, 0].copy()
    cpred = pred[:, :, 0].copy()
    ctrue[ctrue < 255] = 1
    cpred[ctrue < 255] = 1
    right = ctrue[ctrue == cpred]
    hits += np.size(right[right == 1])
    neg_cor += np.size(right[right > 250])
    wrong = ctrue[ctrue != cpred]
    fa_alm += np.size(wrong[wrong > 250])
    miss += np.size(wrong[wrong == 1])
    sz += np.size(ctrue)
    expCor = ((hits + miss) * (hits + fa_alm) +
              (neg_cor + miss) * (neg_cor + fa_alm)) / sz
    hss = ((hits + neg_cor) - expCor) / ((hits + neg_cor) - expCor)
    return hss


# 3 偏差求HSS,例如差距50以上
def biasHss(pred, true, bias):
    # 超过偏差值认为是不等，其中虚警为fore>val+bias,漏报val>fore+bias
    #  hits/false alarm/correct neg/miss
    hits, neg_cor, fa_alm, miss, sz = 0, 0, 0, 0, 0
    ctrue = true[:, :, 0].copy().astype(np.int16)
    cpred = pred[:, :, 0].copy().astype(np.int16)
    # 0~80且相差较小
    hit_cond = np.logical_and(np.abs(ctrue - cpred) <= bias, ctrue < 255 - bias)
    miss_cond = np.logical_and(np.abs(ctrue - cpred) <= bias, ctrue > 255 - bias)
    hit_ind = np.where(hit_cond)
    hits = np.size(hit_ind[0])
    miss_ind = np.where(miss_cond)
    miss = np.size(miss_ind[0])
    neg_ind = np.where(ctrue - cpred > bias)
    neg_cor = np.size(neg_ind[0])
    # fig=plt.figure()
    # plt.imshow(cpred-ctrue)
    # plt.show()
    fa_ind = np.where(cpred - ctrue >bias)
    fa_alm = np.size(fa_ind[0])
    sz += np.size(ctrue)
    expCor = ((hits + miss) * (hits + fa_alm) +
              (neg_cor + miss) * (neg_cor + fa_alm)) / sz
    hss = ((hits + neg_cor) - expCor) / (sz - expCor)
    return hss
#
# def loosehss(pred, true):
#     # 超过偏差值认为是不等，其中虚警为fore>val+bias,漏报val>fore+bias
#     #  hits/false alarm/correct neg/miss
#     hits, neg_cor, fa_alm, miss, sz = 0, 0, 0, 0, 0
#     ctrue = true[:, :, 0].copy().astype(np.int16)
#     cpred = pred[:, :, 0].copy().astype(np.int16)
#     # 0~80且相差较小
#     hit_cond = np.logical_and(np.abs(ctrue<255, cpred < 255))
#     miss_cond = np.logical_and(np.abs(ctrue - cpred) <= bias, ctrue > 255 - bias)
#     hit_ind = np.where(hit_cond)
#     hits = np.size(hit_ind[0])
#     miss_ind = np.where(miss_cond)
#     miss = np.size(miss_ind[0])
#     neg_ind = np.where(ctrue - cpred > bias)
#     neg_cor = np.size(neg_ind[0])
#     # fig=plt.figure()
#     # plt.imshow(cpred-ctrue)
#     # plt.show()
#     fa_ind = np.where(cpred - ctrue >bias)
#     fa_alm = np.size(fa_ind[0])
#     sz += np.size(ctrue)
#     expCor = ((hits + miss) * (hits + fa_alm) +
#               (neg_cor + miss) * (neg_cor + fa_alm)) / sz
#     hss = ((hits + neg_cor) - expCor) / (sz - expCor)
#     return hss


def image_read(file_path):
    image = imread(file_path)
    return image


# forecastPath = '/dpdata/Forecast1004/'
# valPath = '/dpdata/val1/'

forecastPath = 'd:/val1_0924/val1_0924/'
valPath = 'd:/val1/val1/'

dirs = os.listdir(forecastPath)

floss = 0.
cnt=0
for dir in dirs:
    cnt+=1
    # subfiles=os.listdir(forecastPath+dir)
    for i in range(6):
        # f001->031
        foreimg = image_read((forecastPath + dir + '/' + dir + '_' + 'f00%d.png') % (i + 1))
        valname = (valPath + dir + '/' + dir + '_' + '%03d.png') % (30 + 5*i + 1)
        valimg = image_read(valname)
        # floss+=(absLoss(foreimg,valimg)/6)
        # floss += (directHss(foreimg, valimg) / 6)
        floss += (biasHss(foreimg, valimg,50) / 6)
    print(floss/cnt)
print(floss/cnt)

