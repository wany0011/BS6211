"""
find out statistics of images
"""
__author__ = "Patrick Wan"
__Date__ = "16-Sep-2022"
__version__ = "0.0.1"


import cv2
# importing library for plotting
from matplotlib import pyplot as plt

#find out distribution of Mask-pixels
def mask_hist(file):
    # reads an input image
    img = cv2.imread(file, 0)
    # find frequency of pixels in range 0-255
    plt.hist(img.ravel(), 256, [0, 256])
    # show the plotting graph of an image
    plt.show()


if __name__ == '__main__':
    mask_hist(file='./sample/'+'01_mask.png')