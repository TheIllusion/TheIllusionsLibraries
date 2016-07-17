import cv2
import glob
import re
import os

# get list of list files
all_files = glob.glob( '*.*' )

rotate_degrees = [90, 180, 270]

# for each item
i = 1
for image_filename in all_files:
    name = image_filename[0:]

    if ((re.search(".jpg", image_filename) or
         re.search(".JPEG", image_filename) or
         re.search(".JPG", image_filename) or
         re.search(".jpeg", image_filename) or
         re.search(".png", image_filename) or
         re.search(".PNG", image_filename)) and
        ((not re.search("90degree_", image_filename)) and
         (not re.search("180degree_", image_filename)) and
         (not re.search("270degree_", image_filename)))):

        print 'File name:', name

        if not os.path.exists('90degree'):
            os.mkdir('90degree')
        if not os.path.exists('180degree'):
            os.mkdir('180degree')
        if not os.path.exists('270degree'):
            os.mkdir('270degree')

        for rotate_degree in rotate_degrees:
            img = cv2.imread(image_filename, 1)  
	    if not (img == None):
                rows = img.shape[0]
                cols = img.shape[1]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_degree, 1)
                dst = cv2.warpAffine(img, M, (cols, rows))
                new_filename = str(rotate_degree) + 'degree/' + str(rotate_degree) + 'degree_' + name
                cv2.imwrite(new_filename, dst)
                # cv2.imshow('Frame', dst)
                # cv2.waitKey(1000)
