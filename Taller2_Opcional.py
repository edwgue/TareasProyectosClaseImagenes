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

    def filtering(self, imagegr, ang1, deltaang1):
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
        angle_mask[int(half_size), int(half_size)] = 1

         # filtering via FFT

        mask = angle_mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        return image_filtered, mask

    def STDN (self, N, umbral, image):

        Tam_ventana = N
        image_filtered_1 = image
        umbral1 = umbral
        image_filtered_1_M1 = cv2.blur(image_filtered_1, (Tam_ventana, Tam_ventana))
        image_filtered_1_M2 = cv2.blur(np.power(image_filtered_1, 2), (Tam_ventana, Tam_ventana))
        image_filtered_1_STD = np.sqrt(image_filtered_1_M2 - np.power(image_filtered_1_M1, 2))
        image_filtered_1_STD /= np.max(image_filtered_1_STD)
        mask_filtered_1_M1_STD = image_filtered_1_STD > umbral1

        return mask_filtered_1_M1_STD


if __name__ == '__main__':

    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objetothetafilter = thetaFilter(image_gray)
    angtheta = int(input("digite el ángulo theta menor a 40:  "))
    angdeltatheta = int(input("digite el ángulo deltatheta menor a 20:  "))

    image_filtered_1, mask_1 = objetothetafilter.filtering(image_gray, angtheta, angdeltatheta)
    image_filtered_2, mask_2 = objetothetafilter.filtering(image_gray, 80, 20)
    image_filtered_3, mask_3 = objetothetafilter.filtering(image_gray, 108, 20)
    image_filtered_4, mask_4 = objetothetafilter.filtering(image_gray, 144, 20)
    image_filtered_5, mask_5 = objetothetafilter.filtering(image_gray, 182, 20)
    image_filtered_6, mask_6 = objetothetafilter.filtering(image_gray, 220, 20)
    image_filtered_7, mask_7 = objetothetafilter.filtering(image_gray, 256, 20)
    image_filtered_8, mask_8 = objetothetafilter.filtering(image_gray, 292, 20)
    image_filtered_9, mask_9 = objetothetafilter.filtering(image_gray, 320, 20)
    image_filtered_10, mask_10 = objetothetafilter.filtering(image_gray, 355, 20)

    N = 11
    umbral = 0.5
    Mask_image_filtered_1_STD = objetothetafilter.STDN(N, umbral, image_filtered_1)
    Mask_image_filtered_2_STD = objetothetafilter.STDN(N, umbral, image_filtered_2)
    Mask_image_filtered_3_STD = objetothetafilter.STDN(N, umbral, image_filtered_3)
    Mask_image_filtered_4_STD = objetothetafilter.STDN(N, umbral, image_filtered_4)
    Mask_image_filtered_5_STD = objetothetafilter.STDN(N, umbral, image_filtered_5)
    Mask_image_filtered_6_STD = objetothetafilter.STDN(N, umbral, image_filtered_6)
    Mask_image_filtered_7_STD = objetothetafilter.STDN(N, umbral, image_filtered_7)
    Mask_image_filtered_8_STD = objetothetafilter.STDN(N, umbral, image_filtered_8)
    Mask_image_filtered_9_STD = objetothetafilter.STDN(N, umbral, image_filtered_9)
    Mask_image_filtered_10_STD = objetothetafilter.STDN(N, umbral, image_filtered_10)


    rebuilding_image = (image_filtered_1 * mask_1.astype(np.float64)) + (image_filtered_2 * mask_2.astype(np.float64)) + (image_filtered_3 * mask_3.astype(np.float64)) + (image_filtered_4 *
                        mask_4.astype(np.float64)) + (image_filtered_5 * mask_5.astype(np.float64)) + (image_filtered_6 * mask_6.astype(np.float64)) + (image_filtered_7 * mask_7.astype(np.float64)) + (image_filtered_8
                        * mask_8.astype(np.float64)) + (image_filtered_9 * mask_9.astype(np.float64)) + (image_filtered_10 * mask_10.astype(np.float64))



    cv2.imshow("Original image", image)
    cv2.imshow("Filter frequency response", 255 * mask_2)
    cv2.imshow("Filtered image", image_filtered_2)
    cv2.imshow("STD image", 255*(Mask_image_filtered_2_STD.astype(np.uint8)))
    cv2.imshow("Rebuilding image", rebuilding_image)
    cv2.waitKey(0)

