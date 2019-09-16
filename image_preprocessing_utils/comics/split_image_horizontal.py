# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob

#INPUT_DIRECTORY = "/Users/Illusion/Temp/input/"
#ANSWER_DIRECTORY = "/Users/Illusion/Temp/output/"
#OUTPUT_DIRECTORY = "/Users/Illusion/Temp/result_concat/"

INPUT_DIRECTORY = "/Users/Illusion/Downloads/test/"
OUTPUT_DIRECTORY_A = "/Users/Illusion/Downloads/split_A/"
OUTPUT_DIRECTORY_B = "/Users/Illusion/Downloads/split_B/"

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIRECTORY_A):
        os.mkdir(OUTPUT_DIRECTORY_A)

    if not os.path.exists(OUTPUT_DIRECTORY_B):
        os.mkdir(OUTPUT_DIRECTORY_B)

    os.chdir(INPUT_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        print 'filename: ', jpg_file

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            os.system("exit")
            #continue

        h, w, c = input_img.shape
        split_img_A = input_img[:, :w/2, :]
        split_img_B = input_img[:, w/2:, :]

        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY_A, jpg_file), split_img_A)
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY_B, jpg_file), split_img_B)

    print "process finished"


