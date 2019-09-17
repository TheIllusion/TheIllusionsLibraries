import cv2
import numpy as np
import time
import os, glob

#HAND_IMG1 = '/Users/Illusion/Temp/rk.jpg'
#ANI_IMG1 = '/Users/Illusion/Temp/epoch166_fake_B.png'

INPUT_DIR = '/Users/Illusion/Documents/Data/palm_data/personal_info_blurring_exp/testset_real/'
OUTPUT_DIR = '/Users/Illusion/Documents/Data/palm_data/personal_info_blurring_exp/multiple_output/'


def bilateral_filter(img):
    bilateral_result = cv2.bilateralFilter(img, d=50, sigmaColor=90, sigmaSpace=80)
    return bilateral_result


# Kaiming He
def guided_filter(img):
    guided_result = cv2.ximgproc.guidedFilter(img, img, 50, 50)
    return guided_result

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    img_files = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        if type(img) is np.ndarray:

            # Bilateral
            start_time = time.time()
            bilateral_result = bilateral_filter(img)
            elapsed_time = time.time() - start_time
            print 'Bilateral filter elapsed_time: ', elapsed_time

            '''
            start_time = time.time()
            guided_result = guided_filter(img)
            elapsed_time = time.time() - start_time
            print 'Guided filter elapsed_time: ', elapsed_time
            '''

            # L0-smoothing
            start_time = time.time()
            l0_result = cv2.ximgproc.l0Smooth(img, None, 0.012, 2.0)
            elapsed_time = time.time() - start_time
            print 'L0 filter elapsed_time: ', elapsed_time

            # Bilateral + L0-smoothing
            start_time = time.time()
            bi_and_l0_result = cv2.ximgproc.l0Smooth(bilateral_result, None, 0.012, 2.0)
            elapsed_time = time.time() - start_time
            print 'L0(input:bilateral) filter elapsed_time: ', elapsed_time

            concated_img = np.hstack((img, bilateral_result, l0_result, bi_and_l0_result))

            #cv2.imshow('result', concated_img)
            #cv2.waitKey()

            cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_file)), concated_img)

        #break

    print 'process end'