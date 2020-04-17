from skimage.measure import compare_ssim
import numpy as np
import math
import cv2

# SSIM CALCULATION FUNC
def get_ssim(img1, img2):
    ssim = compare_ssim(img1, img2, multichannel=True)
    return ssim

if __name__ == "__main__":

    png_file = '/Users/Illusion/Downloads/trump.png'
    jpg_file = '/Users/Illusion/Downloads/trump_default.jpg'

    decoded_png_img = cv2.imread(png_file, cv2.IMREAD_COLOR)
    decoded_jpg_img = cv2.imread(jpg_file, cv2.IMREAD_COLOR)

    ssim = compare_ssim(decoded_png_img, decoded_jpg_img, multichannel=True)

    print ssim

