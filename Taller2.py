import cv2
import numpy as np
import os
import sys

""" FFT based filtering
    python fft_filtering.py <path_to_image> <image_name>
"""

class thetaFilter:
    def __init__(self, imagen):
        self.imagen_gris = imagen
        self.theta = 1
        self.deltatheta = 1
    def set_theta(theta, deltatheta):
        self.theta = theta
        self.deltatheta = deltatheta
    def filtering():
        1

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    objetothetafilter = thetaFilter(image_gray)
    objetothetafilter.set_theta(45, 5)
    objetothetafilter.filtering()


    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)