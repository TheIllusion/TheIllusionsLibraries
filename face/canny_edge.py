# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

# i7-2600k
INPUT_IMAGE_DIRECTORY_PATH = "/media/illusion/ML_Linux/Data/hair_segmentation/original_all/original_all/"
RESULT_IMAGE_DIRECTORY_PATH = "/media/illusion/ML_Linux/Data/hair_segmentation/original_all/cany_edge_original_all/"

if not os.path.exists(RESULT_IMAGE_DIRECTORY_PATH):
    os.mkdir(RESULT_IMAGE_DIRECTORY_PATH)

if __name__ == "__main__":

    os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        img = cv2.imread(jpg_file, cv2.IMREAD_GRAYSCALE)
        if (type(img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        try:
            edges = cv2.Canny(img, 20, 100)
        except:
            print "Exception occurred in " + jpg_file
            continue

        invert = (255 - edges)

        cv2.imwrite(RESULT_IMAGE_DIRECTORY_PATH + jpg_file, invert)



