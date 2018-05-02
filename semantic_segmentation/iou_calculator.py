# this code calculates mean iou between ground truth images and feedforward result images

import os
import glob
import cv2
import numpy as np

PIXEL_THRESHOLD = 130
#PIXEL_THRESHOLD = 160

# paths (filenames in GT_IMAGE_DIRECTORY and FEEDFORWARD_IMAGE_DIRECTORY must be the same)
#GT_IMAGE_DIRECTORY = '/Users/Illusion/Temp/seg_test/seg_gt/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Temp/seg_test/seg_modified/'

# Custom Network (250x250)
# without cloth
GT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_test_set/resized_250_250_for_custom_nn/gt_image_without_cloth/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/custom_net_v1/forward_result/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/custom_net_v1_without_augmentation/results/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until_0823_and_lfw_aug/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until_0823_aug/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/custom_unet_and_tiramisu/tiramisu_zero_centr_lr_0003_iter_690000/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/custom_unet_and_tiramisu/tiramisu_lfw_added_zero_centr_lr_0_0003/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Downloads/tiramisu_lfw_added_zero_centr_lr_0_0002_iter2760000_ensembels_with_eval/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/result_deeplab_v3/'
FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/result_deeplab_v3_ensemble_6_models/'

# pink to blue by photoshop
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until_0823_background_and_geometry_aug_pink_to_blue/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until_0823_background_aug_pink_to_blue/'

# with cloth
#GT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_test_set/resized_250_250_for_custom_nn/gt_image/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until_0823_background_and_geometry_aug/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until_0823_background_aug/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until0823_without_aug_with_pink/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/forward_result_until0823_aug_with_pink/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Downloads/forward_result/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/hair_semantic_segmentation_pix2pix_without_GAN_until0911_lfw_aug/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/hair_semantic_segmentation_pix2pix_with_gan_until0911_lfw_aug/'
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/hair_semantic_segmentation_pix2pix_without_GAN_until0911_lfw_aug_epoch_9/'

# unet pix2pix (256x256)
'''
GT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/gt_image_without_cloth/'
#with gan
FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/hair_semantic_segmentation_pix2pix/'
#withoug gan
#FEEDFORWARD_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/feedforward_result/hair_semantic_segmentation_pix2pix_without_GAN/'
'''

# Dictionary of answer colors in BGR. Values must be thresholded to 0 or 1.

answer_classes = {}

# hair (red)
answer_classes['hair'] = [0, 0, 1]

# face and skin (green)
answer_classes['face_and_skin'] = [0, 1, 0]

# background (blue)
answer_classes['background'] = [1, 0, 0]

# clothe (pink)
#answer_classes['cloth'] = [1, 0, 1]

# input size
INPUT_GT_IMAGE_SIZE_WIDTH = 256
INPUT_GT_IMAGE_SIZE_HEIGHT = 256

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

    # change the black pixels(unknown) to background pixels
    # iterate through pixels
    black_pixels_mask = np.all(result_img == (0, 0, 0), axis=-1)
    result_img[black_pixels_mask] = tuple(answer_classes['background'])

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

iou_for_answer_classes = {}

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

        #saving the sum of iou for each class
        if not (each_class_name in iou_for_answer_classes.keys()):
            iou_for_answer_classes[each_class_name] = intersection_over_union
        else:
            iou_for_answer_classes[each_class_name] = iou_for_answer_classes[each_class_name] + intersection_over_union

        iou_for_answer_classes[each_class_name]

        print 'intersection over union = ', str(intersection_over_union)
        iou_sum = iou_sum + intersection_over_union

    return iou_sum / len(answer_classes)

if __name__ == "__main__":

    filenames = load_file_names()

    iou_total = 0
    accuracy_total = 0

    check_loop_cnt = 0

    for filename in filenames:

        print 'Filename: ', filename

        # load the pair of ground truth and feedforward result image
        gt_img, result_img = load_pair_images(filename)

        if ((type(gt_img) is not np.ndarray) or ((type(result_img) is not np.ndarray))):
            print filename + ' load failed!'
            break

        if (gt_img.shape[1] is not INPUT_GT_IMAGE_SIZE_WIDTH) or (gt_img.shape[0] is not INPUT_GT_IMAGE_SIZE_HEIGHT):
            gt_img = cv2.resize(gt_img,
                       (INPUT_GT_IMAGE_SIZE_WIDTH, INPUT_GT_IMAGE_SIZE_HEIGHT), cv2.INTER_CUBIC)

        if (result_img.shape[1] is not INPUT_GT_IMAGE_SIZE_WIDTH) or (result_img.shape[0] is not INPUT_GT_IMAGE_SIZE_HEIGHT):
            result_img = cv2.resize(result_img,
                       (INPUT_GT_IMAGE_SIZE_WIDTH, INPUT_GT_IMAGE_SIZE_HEIGHT), cv2.INTER_CUBIC)

        # thresholding pixel values to 0 or 1
        gt_img, result_img = thresholding_images(gt_img, result_img)

        # save the thresholded image. debug purposes only.
        #cv2.imwrite('thresholded_' + filename, result_img*255)

        # calculate overall pixel accuracy
        accuracy = calculate_accuracy(gt_img, result_img)

        # calculate intersection over union
        iou = calculate_iou(gt_img, result_img)

        accuracy_total = accuracy_total + accuracy
        iou_total = iou_total + iou

        check_loop_cnt = check_loop_cnt + 1

        print 'Mean IoU: ', iou
        print 'Pixel Accuracy: ', accuracy
        print '#################################################'

    print 'Number of files: ', str(check_loop_cnt)

    # iterate through classes (eg. face, hair, background ... )
    for each_class_name in iou_for_answer_classes.keys():
        print 'IoU[' + each_class_name + ']: ', iou_for_answer_classes[each_class_name] / len(filenames)

    print 'Class IoU = ', float(iou_total) / len(filenames)
    print 'Per-Pixel Accuracy = ', float(accuracy_total) / len(filenames)
    print 'process finished'