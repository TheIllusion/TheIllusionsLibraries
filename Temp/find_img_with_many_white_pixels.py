# -*- coding: utf-8 -*-

import numpy as np
import glob
import os
import cv2
import shutil

WORKING_DIRECTORY = '/Users/Illusion/Temp/data1/dataset/gsshop/바보사랑/패션잡화_의류/패션잡화/'
DESTINATION_DIRECTORY = '/Users/Illusion/Temp/user_images/'

WHILE_PIXEL_PORTION_THRESHOLD_PERCENT = 15

WHITE_PIXELS = [255, 255, 255]

if not os.path.exists(DESTINATION_DIRECTORY):
    os.mkdir(DESTINATION_DIRECTORY)

img_list = glob.glob(WORKING_DIRECTORY + '*.jpg')

loop_count = 0

for img in img_list:
    img_cv = cv2.imread(img)

    if type(img_cv) is not np.ndarray:
        print 'file open error:', img

    # white pixels
    white_pix_idx = (img_cv[..., 0] == 255) & (img_cv[..., 1] == 255) & (img_cv[..., 2] == 255)
    white_pix_cnt = np.count_nonzero(white_pix_idx)
    other_pix_cnt = np.count_nonzero(np.invert(white_pix_idx))
    white_pix_portion = 100 * float(white_pix_cnt) / (white_pix_cnt + other_pix_cnt)

    if white_pix_portion < WHILE_PIXEL_PORTION_THRESHOLD_PERCENT:
        # this might be an user image
        shutil.copyfile(img, os.path.join(DESTINATION_DIRECTORY, os.path.basename(img)))

    loop_count += 1

    if loop_count % 100 == 0:
        print 'loop_count:', loop_count
        print 'white_pix_cnt =', white_pix_cnt
        print 'other_pix_cnt =', other_pix_cnt
        print 'white_pix_portion =', white_pix_portion

