
# Data Augmentation using background replacement technique
# BACKGROUND IMAGE WILL BE REPLACED BY the IMAGES IN 'BACKGROUND_IMAGE_DIR'

import os, glob
import cv2
import numpy as np
import random

# The Filenames of original images and semantic maps must be the same
TRAINING_IMAGE_DIR = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/lfw/lfw_funneled_jpg_images/'
TRAINING_IMAGE_SEMANTIC_MAPS_DIR = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/lfw/parts_lfw_funneled_gt_jpg_images/'

BACKGROUND_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/background_imgs/refined/'

# Result Images
#RESULT_AUGMENTED_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/result_imgs/'
#RESULT_SEMANTIC_MAPS_DIR = '/Users/Illusion/Documents/Data/background_augmentation/result_semantic_maps/'

RESULT_AUGMENTED_IMAGE_DIR = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/lfw/lfw_funneled_jpg_images_background_augmented/'
RESULT_SEMANTIC_MAPS_DIR = '/Users/Illusion/Documents/Data/hair_semantic_segmentation/lfw/lfw_funneled_jpg_images_background_augmented_sem_maps/'

# Number of desired images per input image (background image will be chosen randomly from 'BACKGROUND_IMAGE_DIR')
NUMBER_OF_AUGMENTED_IMAGES_PER_INPUT = 20

# Dictionary for background colors in BGR. Values must be thresholded to 0 or 1.
background_classes = {}

# background (blue)
background_classes['background'] = [1, 0, 0]

# cloth (pink)
#background_classes['pink'] = [1, 0, 1]

PIXEL_THRESHOLD = 127

if not os.path.exists(RESULT_AUGMENTED_IMAGE_DIR):
    os.mkdir(RESULT_AUGMENTED_IMAGE_DIR)

if not os.path.exists(RESULT_SEMANTIC_MAPS_DIR):
    os.mkdir(RESULT_SEMANTIC_MAPS_DIR)

def load_file_names(full_path):

    try:
        os.chdir(full_path)
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

    sem_map = cv2.imread(TRAINING_IMAGE_SEMANTIC_MAPS_DIR + filename, cv2.IMREAD_COLOR)

    result_img = cv2.imread(TRAINING_IMAGE_DIR + filename, cv2.IMREAD_COLOR)

    return sem_map, result_img

def thresholding_images(sem_map_original):

    sem_map = sem_map_original.copy()

    # thresholding sem_map
    lower_valued_color_elements = sem_map < PIXEL_THRESHOLD
    higher_valued_color_elements = np.invert(lower_valued_color_elements)
    sem_map[lower_valued_color_elements] = 0
    sem_map[higher_valued_color_elements] = 1

    return sem_map

##########################################################################################
# load the filenames
input_img_filenames = load_file_names(TRAINING_IMAGE_SEMANTIC_MAPS_DIR)
background_img_filenames = load_file_names(BACKGROUND_IMAGE_DIR)
##########################################################################################

def get_a_random_img_from_background_imgs():

    file_count = len(background_img_filenames)

    random_pick = random.randrange(0, file_count)

    filename = background_img_filenames[random_pick]

    #print 'background img filename: ', filename

    background_img = cv2.imread(BACKGROUND_IMAGE_DIR + filename, cv2.IMREAD_COLOR)
    if type(background_img) is not np.ndarray:
        print 'loading the background img has failed!'
        os.system("exit")

    return background_img

def augment_background_images(sem_map, sem_map_original, input_img, input_img_filename):

    sem_background_area = np.full((sem_map.shape[0], sem_map.shape[1]), fill_value=False, dtype=bool)

    # save the original image
    cv2.imwrite(RESULT_AUGMENTED_IMAGE_DIR + input_img_filename, input_img)

    # save the semantic map for convenience
    cv2.imwrite(RESULT_SEMANTIC_MAPS_DIR + input_img_filename, sem_map_original)

    for each_class_name in background_classes.keys():

        # iterate through pixels
        # background pixels will be marked as True
        for i in xrange(sem_map.shape[0]):
            for j in xrange(sem_map.shape[1]):

                # detect the class region in ground truth image
                if all(sem_map[i][j] == background_classes[each_class_name]):
                    sem_background_area[i][j] = True

        for i in range(0, NUMBER_OF_AUGMENTED_IMAGES_PER_INPUT, 1):
            # get a background image that is randomly chosen from BACKGROUND_IMAGE_DIR
            background_img = get_a_random_img_from_background_imgs()

            # resize the background image
            background_img = cv2.resize(background_img, (sem_map.shape[1], sem_map.shape[0]), interpolation=cv2.INTER_CUBIC)

            # replace the background pixels
            input_img[sem_background_area] = background_img[sem_background_area]

            # save the augmented image
            cv2.imwrite(RESULT_AUGMENTED_IMAGE_DIR + input_img_filename[:-4] + '_augmented_' + str(i) + '.jpg', input_img)

            # save the semantic map for convenience
            cv2.imwrite(RESULT_SEMANTIC_MAPS_DIR + input_img_filename[:-4] + '_augmented_' + str(i) + '.jpg', sem_map_original)

if __name__ == "__main__":

    idx = 0

    for input_img_filename in input_img_filenames:

        print 'filename: ', input_img_filename

        sem_map_original, input_img = load_pair_images(input_img_filename)

        if (type(sem_map_original) is not np.ndarray) or (type(input_img) is not np.ndarray):
            print 'Image loading failed'
            break

        # thresholding pixel values to 0 or 1
        sem_map = thresholding_images(sem_map_original)

        augment_background_images(sem_map, sem_map_original, input_img, input_img_filename)

        if idx % 100 == 0:
            print 'idx=', str(idx)

        idx = idx + 1
    print 'process finished'