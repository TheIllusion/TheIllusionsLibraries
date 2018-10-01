# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob

INPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/original_all/"
ANSWER_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/seg_result_until_20170911/"
OUTPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/background_erased_images/"

HIGH_PIXEL_THRESHOLD = 200
LOW_PIXEL_THRESHOLD = 50

loop_idx = 0

if __name__ == "__main__":

    os.chdir(ANSWER_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file[:-4] + '.jpg', cv2.IMREAD_UNCHANGED)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        answer_img = cv2.imread( ANSWER_DIRECTORY + jpg_file, cv2.IMREAD_UNCHANGED)
        if (type(answer_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        # apply thresholding to all pixels
        lower_valued_color_elements = answer_img < LOW_PIXEL_THRESHOLD
        higher_valued_color_elements = answer_img > HIGH_PIXEL_THRESHOLD
        answer_img[lower_valued_color_elements] = 0
        answer_img[higher_valued_color_elements] = 255

        # background
        blue_pixels_idx = (answer_img[..., 0] == 255) & (answer_img[..., 1] == 0) & (answer_img[..., 2] == 0)

        # cloth
        pink_pixels_idx = (answer_img[..., 0] == 255) & (answer_img[..., 1] == 0) & (answer_img[..., 2] == 255)

        #print 'blue_pixels_idx =', blue_pixels_idx

        result_img = input_img.copy()

        # make background pixels to white
        result_img[blue_pixels_idx] = 255

        # make cloth pixels to white
        result_img[pink_pixels_idx] = 255

        '''        
        concated_img = np.hstack((input_img, result_img))
        cv2.imshow("result", concated_img)
        cv2.waitKey()
        '''

        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, jpg_file), result_img)

        loop_idx += 1

        if loop_idx % 100  == 0:
            print 'loop count =', loop_idx


