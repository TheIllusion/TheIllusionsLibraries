# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

INPUT_DIRECTORY = "/Users/Illusion/Temp/testA/"
ANSWER_DIRECTORY = "/Users/Illusion/Temp/testB/"
OUTPUT_DIRECTORY = "/Users/Illusion/Temp/"

if __name__ == "__main__":

    os.chdir(ANSWER_DIRECTORY)
    jpg_files = glob.glob( '*.ppm' )

    for jpg_file in jpg_files:

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file[:-4] + '.jpg', cv2.IMREAD_COLOR)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        answer_img = cv2.imread( ANSWER_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(answer_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        red_pixels = answer_img[..., 2] == 255
        not_red_pixels = np.invert(red_pixels)
        answer_img[not_red_pixels] = input_img[not_red_pixels]

        #try:
        concated_img = np.hstack((input_img, answer_img))

        cv2.imwrite(OUTPUT_DIRECTORY + 'trn_' + jpg_file, concated_img)
        #except:
         #   print 'error occurred. skip this image'



