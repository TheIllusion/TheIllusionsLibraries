#Check whether if the input images contain pink lines or not

import numpy as np
import os
import glob
import cv2

img_path = '/Users/Illusion/Documents/Data/palm_data/NEW_DATA_2017/3rd_processed/sm'

os.chdir(img_path)

jpg_files = glob.glob( '*.jpg' )

MINIMUM_THRESHOLD_FOR_PIXEL = 150
MINIMUM_WHITE_PIXELS_COUNT = 150

for jpg in jpg_files:
    #Load image as grayscale
    img = cv2.imread(jpg, 0)

    if(type(img) is not np.ndarray):
        print jpg + ' load failed!'
        break

    high_values_indices = img > MINIMUM_THRESHOLD_FOR_PIXEL

    if  np.sum(high_values_indices) < MINIMUM_WHITE_PIXELS_COUNT:
        print 'White pixels = ', np.sum(high_values_indices)
        print 'Weired file = ', jpg
        print '---------------------------------------------'

    #print 'White pixels = ', np.sum(high_values_indices)
    #print 'Weired file = ', jpg
    #print '---------------------------------------------'

