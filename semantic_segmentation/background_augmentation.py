
# Data Augmentation using background replacement technique
# BACKGROUND IMAGE WILL BE REPLACED BY IMAGES IN 'BACKGROUND_IMAGE_DIR'

import os, glob
import cv2

# Filenames of original images and semantic maps must be the same
TRAINING_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/input_imgs/'
TRAINING_IMAGE_SEMANTIC_MAPS_DIR = '/Users/Illusion/Documents/Data/background_augmentation/semantic_maps/'

BACKGROUND_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/background_imgs/social_events/'
RESULT_AUGMENTED_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/result_imgs/'

# Number of desired images per input image (background image will be chosen randomly from 'BACKGROUND_IMAGE_DIR')
NUMBER_OF_AUGMENTED_IMAGES_PER_INPUT = 5

# Dictionary for background colors in BGR. Values must be thresholded to 0 or 1.
background_classes = {}

# background (blue)
background_classes['background'] = [0, 0, 1]

# cloth (pink)
background_classes['pink'] = [1, 0, 1]

PIXEL_THRESHOLD = 127

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

if __name__ == "__main__":

    input_img_filenames = load_file_names(TRAINING_IMAGE_DIR)
    background_img_filenames = load_file_names(BACKGROUND_IMAGE_DIR)

    for input_img in input_img_filenames:
        sem_map, input_img = load_pair_images(input_img)

    print 'process finished'