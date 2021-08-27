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

    def filtering(imagen, imagegr, ang1, deltaang1):
        theta = ang1
        delta_theta = deltaang1

        # fft visualization
        image_gray = imagegr
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        # pre-computations
        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        # angle based filter mask
        angle_mask = np.zeros_like(image_gray)
        angle = 180 * np.arctan2(row_iter - half_size, col_iter - half_size)/np.pi + 180

        idx_angle_p1 = angle < theta + delta_theta
        idx_angle_p2 = angle > theta - delta_theta
        idx_angle_1 = np.bitwise_and(idx_angle_p1,idx_angle_p2)

        idx_angle_p11 = angle < theta + 180 + delta_theta
        idx_angle_p21 = angle > theta + 180 - delta_theta
        idx_angle_2 = np.bitwise_and(idx_angle_p11, idx_angle_p21)
        idx_angle = np.bitwise_or(idx_angle_1, idx_angle_2)

        angle_mask[idx_angle] = 1

 # filtering via FFT
    mask = angle_mask   # can also use high or band pass mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    cv2.imshow("Original image", image)
    cv2.imshow("Filter frequency response", 255 * mask)
    cv2.imshow("Filtered image", image_filtered)
    cv2.waitKey(0)


if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objetothetafilter = thetaFilter(image_gray)
    angtheta = int(input("digite el ángulo theta "))
    angdeltatheta = int(input("digite el ángulo theta "))
    objetothetafilter.filtering(image_gray, angtheta, angdeltatheta)
    objetothetafilter.set_theta(45, 5)


    cv2.imshow("image", image_gray)
    cv2.waitKey(0)