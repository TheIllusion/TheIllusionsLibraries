# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

# i7-2600k
#INPUT_IMAGE_DIRECTORY_PATH = "/media/illusion/ML_Linux/Data/hair_segmentation/original_all/original_all/"
#RESULT_IMAGE_DIRECTORY_PATH = "/media/illusion/ML_Linux/Data/hair_segmentation/original_all/cany_edge_original_all/"

# Macbook Pro
INPUT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Data/gans_for_video/mug_concat_custom/happiness/"
RESULT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Data/gans_for_video/mug_concat_custom/happiness_canny_edge/"

if not os.path.exists(RESULT_IMAGE_DIRECTORY_PATH):
    os.mkdir(RESULT_IMAGE_DIRECTORY_PATH)

if __name__ == "__main__":

    os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        img = cv2.imread(jpg_file, cv2.IMREAD_COLOR)
        if (type(img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)

        cv2.imwrite(RESULT_IMAGE_DIRECTORY_PATH + jpg_file, img_blur)
