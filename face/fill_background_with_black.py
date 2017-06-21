# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
#import copy

INPUT_DIRECTORY = "/Users/Illusion/Documents/Nail_Art_Change/"
#ANSWER_DIRECTORY = "/Users/Illusion/Temp/testB/"
OUTPUT_DIRECTORY = "/Users/Illusion/Documents/Nail_Art_Change/"

if __name__ == "__main__":

    os.chdir(INPUT_DIRECTORY)
    jpg_files = glob.glob( '*sem.png' )

    for jpg_file in jpg_files:

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file[:-4] + '.png', cv2.IMREAD_COLOR)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        red_pixels = input_img[..., 2] == 255
        not_red_pixels = np.invert(red_pixels)
        input_img[not_red_pixels] = (0, 0, 0)

        cv2.imwrite(OUTPUT_DIRECTORY + 'out_' + jpg_file, input_img)
        #except:
         #   print 'error occurred. skip this image'



