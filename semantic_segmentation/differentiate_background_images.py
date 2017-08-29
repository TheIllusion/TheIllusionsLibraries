# this code is to differentiate background images from randomly collected images using face detector

import os, glob
import dlib
from skimage import io

# Directory settings
BACKGROUND_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/background_imgs/social_events/'
RESULT_AUGMENTED_IMAGE_DIR = '/Users/Illusion/Documents/Data/background_augmentation/background_imgs/refined/'

detector = dlib.get_frontal_face_detector()

def load_file_names():
    try:
        os.chdir(BACKGROUND_IMAGE_DIR)
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

if __name__ == '__main__':

    filenames = load_file_names()

    idx = 0
    for filename in filenames:

        img = io.imread(filename)

        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = detector(img, 1)

        '''
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
        '''

        # face is not detected
        # regard this image as a background image and save it to the result directory
        if len(dets) == 0:
            io.imsave(RESULT_AUGMENTED_IMAGE_DIR + filename, img)

        idx = idx + 1
        if idx % 100 == 0:
            print 'indxt = ', str(idx)