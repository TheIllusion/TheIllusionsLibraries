import os, glob
import sys
import cv2

# test purposes only
'''
ORIGINAL_IMAGE_DIRECTORY = '/Users/Illusion/Temp/pix2pixHD/datasets/cityscapes/train_label'
sys.argv.append(ORIGINAL_IMAGE_DIRECTORY)
'''

DEST_WIDTH = 800
DEST_HEIGHT = 800

if __name__ == "__main__":
    print 'hi'
    print len(sys.argv)
    if len(sys.argv) != 2:
        print 'you need to specify the fullpath of input image directory'
        exit(0)

    directory = sys.argv[1]

    os.chdir(directory)

    dest_path = os.path.join(directory, 'resized_images')

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    img_files = glob.glob('*.png')

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        resized_img = cv2.resize(img, (DEST_WIDTH, DEST_HEIGHT), interpolation=cv2.INTER_CUBIC)

        result_filename = os.path.join(dest_path, img_file)

        cv2.imwrite(result_filename, resized_img)

