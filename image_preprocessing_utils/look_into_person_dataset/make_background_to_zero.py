import cv2
import glob
import os
import numpy as np

# look into person dataset (macbook-pro)

TRAIN_IMAGE_PATH = '/Users/Illusion/Documents/Data_public_set/LIP/TrainVal_images/train_images/'
TRAIN_ANNOTATION_PATH = '/Users/Illusion/Documents/Data_public_set/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/'

VAL_IMAGE_PATH = '/Users/Illusion/Documents/Data_public_set/LIP/TrainVal_images/val_images/'
VAL_ANNOTATION_PATH = '/Users/Illusion/Documents/Data_public_set/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/'

RESULT_IMG_PATH = '/Users/Illusion/Documents/Data_public_set/LIP/background_erased/train'

img_list = glob.glob(os.path.join(TRAIN_IMAGE_PATH, '*.jpg'))

if not os.path.exists(RESULT_IMG_PATH):
    os.mkdir(RESULT_IMG_PATH)

loop_idx = 0

for img_filename in img_list:
    input_img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)

    if type(input_img) is not np.ndarray:
        continue

    annotation_filename = os.path.basename(img_filename)[:-4] + '.png'
    annotation_filepath = os.path.join(TRAIN_ANNOTATION_PATH, annotation_filename)

    annotation_img = cv2.imread(annotation_filepath, cv2.IMREAD_UNCHANGED)

    if type(annotation_img) is not np.ndarray:
        continue

    #print 'filename =', img_filename
    #print 'annotation_filename =', annotation_filepath

    zero_idx = (annotation_img[...] == 0)

    # make background pixels to zero
    input_img[zero_idx] = 0

    # save the result image
    result_filepath = os.path.join(RESULT_IMG_PATH, os.path.basename(img_filename))
    cv2.imwrite(result_filepath, input_img)

    loop_idx = loop_idx + 1

    if loop_idx % 100 == 0:
        print 'loop_idx =', loop_idx
