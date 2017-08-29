# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

INPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/original_all/"
ANSWER_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/seg_result_until_20170823/"
OUTPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/concatenated/"

if __name__ == "__main__":

    os.chdir(ANSWER_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        print 'filename: ', jpg_file

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



