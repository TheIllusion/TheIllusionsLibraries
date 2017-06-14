# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

INPUT_DIRECTORY = "/Users/Illusion/Pictures/Hair_Creation/"
ANSWER_DIRECTORY = "/Users/Illusion/Pictures/Hair_Creation/"
OUTPUT_DIRECTORY = "/Users/Illusion/Pictures/Hair_Creation/concatenated/"

if __name__ == "__main__":

    os.chdir(ANSWER_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        canny_img = cv2.imread( INPUT_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(canny_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        answer_img = cv2.imread( ANSWER_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(answer_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        #try:
        concated_img = np.hstack((canny_img, answer_img))

        cv2.imwrite(OUTPUT_DIRECTORY + 'trn_' + jpg_file, concated_img)
        #except:
         #   print 'error occurred. skip this image'



