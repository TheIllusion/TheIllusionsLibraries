import cv2
import numpy as np
import time
import os, glob

#HAND_IMG1 = '/Users/Illusion/Temp/rk.jpg'
#ANI_IMG1 = '/Users/Illusion/Temp/epoch166_fake_B.png'

INPUT_DIR = '/Users/Illusion/Documents/Data/palm_data/personal_info_blurring_exp/testset_real/'
OUTPUT_DIR = '/Users/Illusion/Documents/Data/palm_data/personal_info_blurring_exp/multiple_output/'

#INPUT_DIR = '/Users/Illusion/Temp/full_color_pix2pix/'
#OUTPUT_DIR = '/Users/Illusion/Temp/full_color_pix2pix_multiple_output/'

#INPUT_DIR = '/Users/Illusion/Temp/ex_color_pix2pix/'
#OUTPUT_DIR = '/Users/Illusion/Temp/ex_color_pix2pix_multiple_output/'

SINGLE_IMG_VIEW_FLAG = False

# FILTER LIST
EDGE_PRESERVING = True
GUIDED = True
BILATERAL = True
L0_SMOOTHING = True
BI_AND_LO = True

# Text
ADD_TEXT = True

font                   = cv2.FONT_HERSHEY_SIMPLEX
location = (10, 30)
fontScale              = 0.7
fontColor              = (255,255,255)
lineType               = 2

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    img_files = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))

    idx = 0

    for img_file in img_files:

        print '---------- img idx: ', idx + 1
        idx += 1

        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        if type(img) is np.ndarray:

            result_imgs = []
            result_imgs.append(img)

            # edge-preserving (opencv v4)
            if EDGE_PRESERVING:
                start_time = time.time()
                # ximgproc::edgePreservingFilter(image, res, 9, 20);
                edge_preserving_result = cv2.ximgproc.edgePreservingFilter(img, d=9, threshold=10)
                if ADD_TEXT:
                    cv2.putText(edge_preserving_result, 'Edge-Preserving Filter', location, font, fontScale, fontColor, lineType)

                result_imgs.append(edge_preserving_result)
                elapsed_time = time.time() - start_time
                print 'Edge-preserving filter elapsed_time: ', elapsed_time

            if GUIDED:
                # Guided (Kaiming He)
                start_time = time.time()
                guided_result = cv2.ximgproc.guidedFilter(img.copy(), img, 8, 500)
                if ADD_TEXT:
                    cv2.putText(guided_result, 'Guided Filter', location, font, fontScale, fontColor, lineType)

                result_imgs.append(guided_result)
                elapsed_time = time.time() - start_time
                print 'Guided filter elapsed_time: ', elapsed_time

            if BILATERAL:
                # Bilateral
                start_time = time.time()
                bilateral_result = cv2.bilateralFilter(img, d=60, sigmaColor=90, sigmaSpace=40)
                if ADD_TEXT:
                    cv2.putText(bilateral_result, 'Bilateral Filter', location, font, fontScale, fontColor, lineType)

                result_imgs.append(bilateral_result)
                elapsed_time = time.time() - start_time
                print 'Bilateral filter elapsed_time: ', elapsed_time

                # stronger
                # start_time = time.time()
                # bilateral_result = cv2.bilateralFilter(img, d=60, sigmaColor=150, sigmaSpace=40)
                # result_imgs.append(bilateral_result)
                # elapsed_time = time.time() - start_time
                # print 'Bilateral filter elapsed_time: ', elapsed_time

                # stronger (using resize)
                # start_time = time.time()
                # h, w, c = img.shape
                # resized_img = cv2.resize(img, (h / 2, w / 2), interpolation=cv2.INTER_CUBIC)
                # bilateral_result = cv2.bilateralFilter(resized_img, d=30, sigmaColor=150, sigmaSpace=40)
                # bilateral_result = cv2.resize(bilateral_result, (h, w), interpolation=cv2.INTER_CUBIC)
                # result_imgs.append(bilateral_result)
                # elapsed_time = time.time() - start_time
                # print 'Bilateral filter elapsed_time: ', elapsed_time

                # stronger
                # start_time = time.time()
                # bilateral_result = cv2.bilateralFilter(img, d=60, sigmaColor=220, sigmaSpace=40)
                # result_imgs.append(bilateral_result)
                # elapsed_time = time.time() - start_time
                # print 'Bilateral filter elapsed_time: ', elapsed_time

                # stronger (using resize)
                # start_time = time.time()
                # h, w, c = img.shape
                # resized_img = cv2.resize(img, (h/2, w/2), interpolation=cv2.INTER_CUBIC)
                # bilateral_result = cv2.bilateralFilter(resized_img, d=30, sigmaColor=220, sigmaSpace=40)
                # bilateral_result = cv2.resize(bilateral_result, (h, w), interpolation=cv2.INTER_CUBIC)
                # result_imgs.append(bilateral_result)
                # elapsed_time = time.time() - start_time
                # print 'Bilateral filter elapsed_time: ', elapsed_time

            if L0_SMOOTHING:
                # L0-smoothing
                l0_lambda = 0.005
                l0_kappa = 1.5
                start_time = time.time()
                l0_result = cv2.ximgproc.l0Smooth(img, None, l0_lambda, l0_kappa)
                if ADD_TEXT:
                    cv2.putText(l0_result, 'L0-smoothing Filter', location, font, fontScale, fontColor, lineType)

                result_imgs.append(l0_result)
                elapsed_time = time.time() - start_time
                print 'L0 filter elapsed_time: ', elapsed_time

            if BI_AND_LO:
                # Bilateral + L0-smoothing
                start_time = time.time()
                bi_and_l0_result = cv2.ximgproc.l0Smooth(bilateral_result, None, l0_lambda, l0_kappa)
                if ADD_TEXT:
                    cv2.putText(bi_and_l0_result, '                + L0-smoothing', location, font, fontScale, fontColor, lineType)
                result_imgs.append(bi_and_l0_result)
                elapsed_time = time.time() - start_time
                print 'L0(input:bilateral) filter elapsed_time: ', elapsed_time

            concat_img = np.hstack(tuple(result_imgs))

            if SINGLE_IMG_VIEW_FLAG:
                cv2.imshow('result', concat_img)
                cv2.waitKey()
            else:
                cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(img_file)), concat_img)

        if SINGLE_IMG_VIEW_FLAG:
            break

    print 'process end'