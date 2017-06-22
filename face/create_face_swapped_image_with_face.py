# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

#ANSWER_DIRECTORY = "/home1/irteamsu/users/rklee/hair_segmentation/exp1/forward_result/"
#INPUT_DIRECTORY = "/home1/irteamsu/users/rklee/hair_segmentation/exp1/forward_result/"
#OUTPUT_DIRECTORY = "/home1/irteamsu/users/rklee/gan/pix2pix/datasets/semantic_to_hair_with_face_svc/train/"

ORIGINAL_DIRECTORY = "/home1/irteamsu/users/rklee/gan/CycleGAN/results/black_to_wine_both_datasets_rotated/latest_test/images/real_A/"
ANSWER_DIRECTORY = "/home1/irteamsu/users/rklee/gan/CycleGAN/results/black_to_wine_both_datasets_rotated/latest_test/images/real_A_thresholded/"
INPUT_DIRECTORY = "/home1/irteamsu/users/rklee/gan/CycleGAN/results/black_to_wine_both_datasets_rotated/latest_test/images/fake_B/"
OUTPUT_DIRECTORY = "/home1/irteamsu/users/rklee/gan/CycleGAN/results/black_to_wine_both_datasets_rotated/latest_test/images/fake_B_face_swapped/"

PIXEL_THRESHOLD_VALUE = 70

if __name__ == "__main__":

    os.chdir(ANSWER_DIRECTORY)
    jpg_files = glob.glob( 'out_*.jpg' )

    for jpg_file in jpg_files:

        original_img = cv2.imread( ORIGINAL_DIRECTORY + jpg_file[4:-4] + '.png', cv2.IMREAD_COLOR)
        if (type(original_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file[4:-4] + '.png', cv2.IMREAD_COLOR)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        answer_img = cv2.imread( ANSWER_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(answer_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        answer_img = cv2.resize(answer_img, (128,128), interpolation=cv2.INTER_CUBIC)

        green_pixels = answer_img[..., 1] > PIXEL_THRESHOLD_VALUE
        #not_red_pixels = np.invert(red_pixels)
        input_img[green_pixels] = original_img[green_pixels]
        #answer_img[red_pixels] = (0, 0, 255)
        
        #try:
        #concated_img = np.hstack((input_img, answer_img))

        cv2.imwrite(OUTPUT_DIRECTORY + jpg_file[4:-4] + '.png', input_img)
        #except:
         #   print 'error occurred. skip this image'

