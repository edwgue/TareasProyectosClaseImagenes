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

    def set_theta(self, theta, deltatheta):
        self.theta = theta
        self.deltatheta = deltatheta

    def filtering(imagen, imagefftshift):

        # fft visualization
        image_gray_fft_shift = imagefftshift
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        # pre-computations
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2 - 1  # here we assume num_rows = num_columns

        # low pass filter mask
        low_pass_mask = np.zeros_like(image_gray)
        freq_cut_off = 0.3  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx_lp] = 1

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    objetothetafilter = thetaFilter(image_gray)
    objetothetafilter.set_theta(45, 5)
    objetothetafilter.filtering(image_gray)







    cv2.imshow("image", image_gray)
    cv2.waitKey(0)