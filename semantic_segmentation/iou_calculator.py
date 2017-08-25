# this code calculates mean iou between ground truth images and feedforward result images

import os
import glob
import cv2
import numpy as np

# paths (filenames in GT_IMAGE_DIRECTORY and FEEDFORWARD_IMAGE_DIRECTORY must be the same)
GT_IMAGE_DIRECTORY = '/Users/Illusion/Temp/seg_test/seg_gt/'
FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Temp/seg_test/seg_modified/'

# Dictionary of answer colors in BGR. Values must be thresholded to 0 or 1.

answer_classes = {}

# hair (red)
answer_classes['hair'] = [0, 0, 1]

# face and skin (green)
answer_classes['face_and_skin'] = [0, 1, 0]

# background (blue)
answer_classes['background'] = [1, 0, 0]

# clothe (pink)
answer_classes['clothe'] = [1, 0, 1]

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

def load_pair_images(filename):
    gt_img = cv2.imread(GT_IMAGE_DIRECTORY + filename, cv2.IMREAD_COLOR)

    result_img = cv2.imread(FEEDFORWARD_IMAGE_DIRECTORY + filename, cv2.IMREAD_COLOR)

    return gt_img, result_img

PIXEL_THRESHOLD = 127

def thresholding_images(gt_img, result_img):

    # thresholding gt_img
    lower_valued_color_elements = gt_img < PIXEL_THRESHOLD
    higher_valued_color_elements = np.invert(lower_valued_color_elements)
    gt_img[lower_valued_color_elements] = 0
    gt_img[higher_valued_color_elements] = 1

    # thresholding result_img
    lower_valued_color_elements = result_img < PIXEL_THRESHOLD
    higher_valued_color_elements = np.invert(lower_valued_color_elements)
    result_img[lower_valued_color_elements] = 0
    result_img[higher_valued_color_elements] = 1

    return gt_img, result_img

def calculate_accuracy(gt_img, result_img):
    bool_array = np.equal(gt_img, result_img)
    #pixel_bool_array = all(bool_array_temp[..., 0])
    unmatched_count = 0
    total_pixels = gt_img.shape[0] * gt_img.shape[1]

    for i in xrange(gt_img.shape[0]):
        for j in xrange(gt_img.shape[1]):
            pixel = bool_array[i][j]
            if pixel[0] == False or pixel[1] == False or pixel[2] == False:
                unmatched_count = unmatched_count + 1

    print 'Unmatched pixel count = ', str(unmatched_count)
    print 'Total pixels = ', str(total_pixels)

    accuracy = float(100) * (total_pixels - unmatched_count) / total_pixels

    return accuracy

def calculate_iou(gt_img, result_img):

    result_area = np.empty([gt_img.shape[0], gt_img.shape[1]], bool)
    gt_area = np.empty([gt_img.shape[0], gt_img.shape[1]], bool)

    iou_sum = 0

    # iterate through classes (eg. face, hair, background ... )
    for each_class_name in answer_classes.keys():

        # iterate through pixels
        for i in xrange(gt_img.shape[0]):
            for j in xrange(gt_img.shape[1]):

                # detect the class region in ground truth image
                if all(gt_img[i][j] == answer_classes[each_class_name]):
                    gt_area[i][j] = True
                else:
                    gt_area[i][j] = False

                # detect the class region in feedforward result image
                if all(result_img[i][j] == answer_classes[each_class_name]):
                    result_area[i][j] = True
                else:
                    result_area[i][j] = False

        print '-------------------------------'
        print 'class = ', each_class_name
        print 'class(value) = ', answer_classes[each_class_name]

        # calculate intersection area
        intersection = np.logical_and(gt_area, result_area)
        intersection_area = np.sum(intersection)
        print 'intersection area = ', str(intersection_area)

        # calculate union area
        union = np.logical_or(gt_area, result_area)
        union_area = np.sum(union)
        print 'union area = ', str(union_area)

        if union_area > 0:
            intersection_over_union = float(100) * intersection_area / union_area
        else:
            intersection_over_union = 100

        print 'intersection over union = ', str(intersection_over_union)
        iou_sum = iou_sum + intersection_over_union

    return iou_sum / len(answer_classes)

if __name__ == "__main__":

    filenames = load_file_names()

    iou_total = 0
    accuracy_total = 0

    for filename in filenames:

        # load the pair of ground truth and feedforward result image
        gt_img, result_img = load_pair_images(filename)

        if ((type(gt_img) is not np.ndarray) or ((type(result_img) is not np.ndarray))):
            print filename + ' load failed!'
            break

        # thresholding pixel values to 0 or 1
        gt_img, result_img = thresholding_images(gt_img, result_img)

        # calculate overall pixel accuracy
        accuracy = calculate_accuracy(gt_img, result_img)

        # calculate intersection over union
        iou = calculate_iou(gt_img, result_img)

        accuracy_total = accuracy_total + accuracy
        iou_total = iou_total + iou

        print 'Filename: ', filename
        print 'Mean IoU: ', iou
        print 'Pixel Accuracy: ', accuracy
        print '#################################################'

    print 'Overall Mean IoU = ', float(iou_total) / len(filenames)
    print 'Overall Pixel Accuracy = ', float(accuracy_total) / len(filenames)
    print 'process finished'