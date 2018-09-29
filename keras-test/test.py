
from PIL import Image
from matplotlib import pyplot as plt

from scipy.misc import imread

def image_read(file_path):
    image = imread(file_path)
    return image



image = image_read('/dpdata/testforecast/RAD_307073425600404/RAD_307073425600404_f001.png')
image2 = image_read('/dpdata/SRAD2018_FORECAST2/RAD_307073425600404/RAD_307073425600404_f001.png')
# image = np.reshape(image, [1, oheight, owidth, 3])
image = Image.fromarray(image)
# image = image.resize([height, width])