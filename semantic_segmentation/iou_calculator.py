# this code calculates mean iou between ground truth images and feedforward result images

import os
import glob
import cv2

# paths (filenames in GT_IMAGE_DIRECTORY and FEEDFORWARD_IMAGE_DIRECTORY must be the same)
GT_IMAGE_DIRECTORY = '/Users/Illusion/Temp/seg_test/seg_gt/'
FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Temp/seg_test/seg_modified/'

# list of answer colors in BGR

# hair (red)
hair = [0, 0, 255]

# face and skin (green)
face_and_skin = [0, 255, 0]

# background (blue)
background = [255, 0, 0]

# clothe (pink)
# clothe = [255, 0, 255]

answer_colors = []
answer_colors.append(hair)
answer_colors.append(face_and_skin)
answer_colors.append(background)

def load_file_names():
    try:
        os.chdir(FEEDFORWARD_IMAGE_DIRECTORY)
    except:
        print 'Failed to change directory. Please check the path.'
        os.system("exit")

    jpg_files = glob.glob('*.jpg')
    JPG_files = glob.glob('*.JPG')
    jpeg_files = glob.glob('*.jpeg')
    JPEG_files = glob.glob('*.JPEG')
    png_files = glob.glob('*.png')
    PNG_files = glob.glob('*.PNG')

    filenames = jpg_files + JPG_files + jpeg_files + JPEG_files + png_files + PNG_files

    return filenames

def load_pair_images():
    cv2.
def calculate_iou():

def calculate_accuracy():

if __name__ == "__main__":

    filenames = load_file_names()

    for file in filenames:
        # load the pair of ground truth and feedforward result image
        gt_img, result_img = load_pair_images()

        iou = calculate_iou(gt_img, result_img)

        accuracy = calculate_accuracy(gt_img, result_img)

        print '------------------------------'
        print 'Filename: ', file
        print 'IOU: ', iou
        print 'Accuracy: ', accuracy

    '''
    for color in answer_colors:
        # calculate iou
    '''

    print 'process finished'