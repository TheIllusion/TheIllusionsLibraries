# -*- coding: utf-8 -*-

import numpy as np
import cv2
#import os
import glob
#import copy

if __name__ == "__main__":

    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        img = cv2.imread(jpg_file, cv2.IMREAD_GRAYSCALE)
        if (type(img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        try:
            edges = cv2.Canny(img, 20, 100)
        except:
            print "Exception occurred in " + jpg_file
            continue

        invert = (255 - edges)

        cv2.imwrite('canny_' + jpg_file, invert)



