from collections import Counter
import cv2
import numpy as np

def count_pixel_dist(img):

    # flatten img to 1d
    img = img.flatten()

    # convert to python list
    img_as_python_list = img.tolist()

    counted_dict = Counter(img_as_python_list)

    # print the distributions of pixel values
    for i in range(256):
        print str(i) + ' :', counted_dict[i]

if __name__ == '__main__':

    image = cv2.imread('/Users/Illusion/Documents/rk_face.jpg', cv2.IMREAD_COLOR)

    if type(image) is not np.ndarray:
        print 'cannot read file'
        exit()

    count_pixel_dist(image)

