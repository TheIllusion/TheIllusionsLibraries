import numpy as np
import sys, os, glob
import cv2

# test purposes only
ORIGINAL_IMAGE_DIRECTORY = '/Users/Illusion/Temp/pix2pixHD/datasets/cityscapes/train_label'
sys.argv.append(ORIGINAL_IMAGE_DIRECTORY)

HISTOGRAM_BIN_BUFFER_SIZE = 50000

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'you need to specify the fullpath of input image directory'
        exit(0)

    directory = sys.argv[1]

    os.chdir(directory)

    img_files = glob.glob('*.png')

    for img_file in img_files:

        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        # pixel values have range from 0 to 255
        histogram_bin = np.zeros(shape=HISTOGRAM_BIN_BUFFER_SIZE, dtype=np.uint8)

        height, width = img.shape
        print 'width:', str(width), 'height:', str(height)

        # iterate through pixels
        for i in range(width):
            for j in range(height):
                histogram_bin[img[j][i]] = histogram_bin[img[j][i]] + 1

        # iterate through bins and print the information
        total_pixels = 0
        total_labels = 0

        for i in range(HISTOGRAM_BIN_BUFFER_SIZE):
            if histogram_bin[i] != 0:
                print 'histogram of', str(i), ':', str(histogram_bin[i])
                total_pixels = total_pixels + histogram_bin[i]
                total_labels = total_labels + 1

        print 'total_pixels: ', str(total_pixels)
        print 'total_labels: ', str(total_labels)
        print '========================================='

        #break