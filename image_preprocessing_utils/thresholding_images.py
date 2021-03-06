import os, glob
import sys
import cv2
import numpy as np

# test purposes only
'''
ORIGINAL_IMAGE_DIRECTORY = '/Users/Illusion/Temp/pix2pixHD/datasets/cityscapes/train_label'
sys.argv.append(ORIGINAL_IMAGE_DIRECTORY)
'''

PIXEL_THRESHOLD = 200

def thresholding_images(input_img):

    # thresholding the img
    lower_valued_color_elements = input_img < PIXEL_THRESHOLD
    higher_valued_color_elements = np.invert(lower_valued_color_elements)
    input_img[lower_valued_color_elements] = 0
    input_img[higher_valued_color_elements] = 255

    return input_img

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'you need to specify the fullpath of input image directory'
        exit(0)

    directory = sys.argv[1]

    os.chdir(directory)

    dest_path = os.path.join(directory, 'thresholded_images')

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    img_files = glob.glob('*.png')

    img_idx = 0

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        thresholded_img = thresholding_images(img)

        result_filename = os.path.join(dest_path, img_file)

        cv2.imwrite(result_filename, thresholded_img)

        img_idx = img_idx + 1
        if img_idx % 100 == 0:
            print 'index: ', str(img_idx)

