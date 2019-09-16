import cv2
import numpy as np
import time
import os, glob

#HAND_IMG1 = '/Users/Illusion/Temp/rk.jpg'
#ANI_IMG1 = '/Users/Illusion/Temp/epoch166_fake_B.png'

INPUT_DIR = '/Users/Illusion/Documents/Data/palm_data/personal_info_blurring_exp/testset_real/'
OUTPUT_DIR = '/Users/Illusion/Documents/Data/palm_data/personal_info_blurring_exp/bilateral_output/'

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    img_files = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        if type(img) is np.ndarray:

            start_time = time.time()

            result = cv2.bilateralFilter(img, d=50, sigmaColor=90, sigmaSpace=80)

            elapsed_time = time.time() - start_time

            concated_img = np.hstack((img, result))

            #cv2.imshow('result', concated_img)

            print 'elapsed_time: ', elapsed_time

            #cv2.waitKey()

            cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_file)), concated_img)

    print 'process end'