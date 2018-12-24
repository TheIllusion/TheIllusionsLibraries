import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('./jpgs/person_mask_png.png', cv2.IMREAD_UNCHANGED)

    if type(img) is np.ndarray:

        # erosion
        #kernel = np.ones((5, 5), np.uint8)
        #eroded_img = cv2.erode(img, kernel, iterations=1)

        # dilation
        kernel = np.ones((5, 5), np.uint8)
        dilated_img = cv2.dilate(img, kernel, iterations=1)

        # averaging
        #blurred_img = cv2.blur(eroded_img, (10, 10))
        blurred_img = cv2.blur(dilated_img, (10, 10))

        #cv2.imshow('original', img)
        #cv2.imshow('result', blurred_img)

        # stack images vertically
        #vis = np.concatenate((img, eroded_img, blurred_img), axis=0)
        vis = np.concatenate((img, dilated_img, blurred_img), axis=0)

        cv2.imshow('results', vis)

        cv2.waitKey()

    print 'process end'