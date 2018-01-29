import os, glob
import sys
import cv2
import numpy as np

# test purposes only
ORIGINAL_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/lfw/parts_lfw_funneled_gt_images'
sys.argv.append(ORIGINAL_IMAGE_DIRECTORY)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'you need to specify the fullpath of input image directory'
        exit(0)

    directory = sys.argv[1]

    os.chdir(directory)

    dest_path = os.path.join(directory, 'jpg_imgs')

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    img_files = glob.glob('*.ppm')

    img_idx = 0

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)

        result_filename = os.path.join(dest_path, img_file[0:-4] + '.jpg')

        cv2.imwrite(result_filename, img)

        img_idx = img_idx + 1
        if img_idx % 100 == 0:
            print 'index: ', str(img_idx)

