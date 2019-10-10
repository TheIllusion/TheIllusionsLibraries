# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob

#INPUT_DIRECTORY = "/Users/Illusion/Temp/input/"
#ANSWER_DIRECTORY = "/Users/Illusion/Temp/output/"
#OUTPUT_DIRECTORY = "/Users/Illusion/Temp/result_concat/"

#INPUT_DIRECTORY = "/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/grey_split/"
#ANSWER_DIRECTORY = "/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/23_colors/"
#OUTPUT_DIRECTORY = "/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/pix2pix_23_colors/"

INPUT_DIRECTORY = "/Volumes/Ext_850Ev/Comico/paired_data_partial/testA/"
ANSWER_DIRECTORY = "/Volumes/Ext_850Ev/Comico/paired_data_partial/testB/"
OUTPUT_DIRECTORY = "/Volumes/Ext_850Ev/Comico/paired_data_partial/testAB_concat/"

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

        answer_img = cv2.imread( ANSWER_DIRECTORY + jpg_file, cv2.IMREAD_COLOR)
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

        png_file = jpg_file[:-4] + '.png'

        cv2.imwrite(OUTPUT_DIRECTORY + 'concat_' + png_file, concated_img)
        #except:
         #   print 'error occurred. skip this image'

    print "process finished"


