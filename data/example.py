# import imread
# import scipy.misc as misc
from scipy.misc import imread
import matplotlib.pyplot as plt

#
# def image_read(file_path):
#     image=imread(file_path)
#     return image
#
#
# im=image_read('/dpdata/SRAdata/RAD_206482464212530/RAD_206482464212530_060.png')
#
# plt.subplot(1, 2, 1)
# plt.imshow(im)
# plt.axis('off')
# plt.show()
import  numpy as np
array=np.ones(shape=[2,3,6])

print(array)
print('================')
array[:,:]=0
print(array)
print('================')
array[0]=3
print(array)
print('================')
array[:,:,:,:]=2
print(array)


