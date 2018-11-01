import numpy as np
import cv2
import os
import glob

#INPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/background_erased_images/"
INPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/seg_result_until_20170911/"

#OUTPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/background_erased_images_with_padding/"
OUTPUT_DIRECTORY = "/Users/Illusion/Documents/Data/hair_semantic_segmentation/official_training_set/seg_result_until_20170911_with_padding/"

if not os.path.exists(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)

def add_padding_aroung_image(inputImage):
    #TEST_IMAGE = '/Users/Illusion/Temp/sample.jpg'
    #inputImage = cv2.imread(TEST_IMAGE, cv2.IMREAD_UNCHANGED)

    if type(inputImage) is not np.ndarray:
        exit(-1)

    paddingAmount = 100

    topBorderWidth = paddingAmount
    bottomBorderWidth = paddingAmount
    leftBorderWidth = paddingAmount
    rightBorderWidth = paddingAmount

    color_of_border = (255, 0, 0)

    # apply resizing (optional)
    inputImage = cv2.resize(input_img, (512, 512))

    outputImage = cv2.copyMakeBorder(
                     inputImage,
                     topBorderWidth,
                     bottomBorderWidth,
                     leftBorderWidth,
                     rightBorderWidth,
                     cv2.BORDER_CONSTANT,
                     value=color_of_border
                  )

    # cv2.imshow("output", outputImage)
    # cv2.waitKey()

    return outputImage


loop_idx = 0

if __name__ == "__main__":

    os.chdir(INPUT_DIRECTORY)
    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        input_img = cv2.imread( INPUT_DIRECTORY + jpg_file[:-4] + '.jpg', cv2.IMREAD_UNCHANGED)
        if (type(input_img) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        result_img = add_padding_aroung_image(input_img)

        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, jpg_file), result_img)

        loop_idx += 1

        if loop_idx % 100  == 0:
            print 'loop count =', loop_idx
