# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

INPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_염색_아르바이트/hair_style_generation/total_datasets/original/"

ANSWER_DIRECTORY = "/Users/Illusion/Documents/Data/hair_염색_아르바이트/hair_style_generation/total_datasets/hair_edited/retro_finger_perm/"

OUTPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_염색_아르바이트/hair_style_generation/total_datasets/hair_edited/retro_finger_perm_concat/"

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    os.chdir(ANSWER_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        print 'filename: ', jpg_file

        canny_img = cv2.imread( INPUT_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(canny_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            os.system("exit")
            #continue

        answer_img = cv2.imread( ANSWER_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(answer_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            os.system("exit")
            #continue

        #try:
        concated_img = np.hstack((canny_img, answer_img))

        cv2.imwrite(OUTPUT_DIRECTORY + 'trn_' + jpg_file, concated_img)
        #except:
         #   print 'error occurred. skip this image'

    print "process finished"


