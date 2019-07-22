from collections import Counter
import cv2
import numpy as np

def count_pixel_dist(img):

    print 'img.shape =', img.shape

    img2 = []

    for ch in img.tolist():
        for p in ch:
            #print tuple(p)
            img2.append(tuple(p))


    counted_dict = Counter(img2)

    print counted_dict
    print 'total pixels =', len(img2)
    print 'num of colors =', len(counted_dict)


if __name__ == '__main__':

    image = cv2.imread('/Users/Illusion/Temp/30_colors/nicoic-03-148-066-02-f-pjw-160804.jpg_00 copy.png', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Users/Illusion/Temp/yonggu-61-003-172-02-f-whk-181213.jpg', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Users/Illusion/Temp/20_color_yonggu-61-003-172-02-f-whk-181213.png', cv2.IMREAD_COLOR)

    if type(image) is not np.ndarray:
        print 'cannot read file'
        exit()

    count_pixel_dist(image)

