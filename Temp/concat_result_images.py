# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob

#INPUT_DIRECTORY = "/Users/Illusion/Temp/input/"
#ANSWER_DIRECTORY = "/Users/Illusion/Temp/output/"
#OUTPUT_DIRECTORY = "/Users/Illusion/Temp/result_concat/"

INPUT_DIRECTORY = "/home1/irteamsu/users/gihyeok/JPPNet_1224/LIP_JPPNet/datasets/examples/images/"
ANSWER_DIRECTORY = "/home1/irteamsu/users/gihyeok/JPPNet_1224/LIP_JPPNet/output/parsing/val/"
OUTPUT_DIRECTORY = "/home1/irteamsu/users/gihyeok/JPPNet_1224/LIP_JPPNet/output/parsing/concat_result/"

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    os.chdir(INPUT_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        print 'filename: ', jpg_file

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            os.system("exit")
            #continue

        png_file = jpg_file[:-4] + '_vis.png'
        answer_img = cv2.imread( ANSWER_DIRECTORY + png_file, cv2.IMREAD_COLOR)
        if (type(answer_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            os.system("exit")
            #continue

        # resize the answer_img to the size of input_img
        if input_img.shape[0] != answer_img.shape[0] or input_img.shape[1] != answer_img.shape[1]:
            print 'resize the image'
            answer_img = cv2.resize(answer_img, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_CUBIC)

        #try:
        concated_img = np.hstack((input_img, answer_img))

        cv2.imwrite(OUTPUT_DIRECTORY + 'concat_' + jpg_file, concated_img)
        #except:
         #   print 'error occurred. skip this image'

    print "process finished"


