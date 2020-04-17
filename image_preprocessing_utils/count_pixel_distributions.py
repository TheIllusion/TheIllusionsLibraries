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

    #image = cv2.imread('/Users/Illusion/Temp/aaa.jpg', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Users/Illusion/Temp/30_colors/nicoic-05-066-104-04-f-knh-161011.jpg_02.jpg', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/pix2pix_full_colors/concat_nicoic-02-023-027-05-f-whk-160417.jpg_04.jpg', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Users/Illusion/Temp/yonggu-61-003-172-02-f-whk-181213.jpg', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Users/Illusion/Temp/20_color_yonggu-61-003-172-02-f-whk-181213.png', cv2.IMREAD_COLOR)
    #image = cv2.imread('/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/color_split/nicoic-03-149-066-03-f-pjw-160804.jpg_02.jpg')
    #image = cv2.imread('/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/pix2pix_23_colors/concat_nicoic-02-026-027-08-f-whk-160524.jpg_00.png')
    #image = cv2.imread('/Users/Illusion/Temp/concat_nicoic-03-148-066-01-f-pjw-160803.jpg_01.png')
    #image = cv2.imread('/Users/Illusion/Temp/15_without_background.png')
    #image = cv2.imread('/Users/Illusion/Temp/26_with_e.png')
    image = cv2.imread('/Users/Illusion/Temp/small.png')

    if type(image) is not np.ndarray:
        print 'cannot read file'
        exit()

    count_pixel_dist(image)

