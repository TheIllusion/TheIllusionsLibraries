import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('./mask.png', cv2.IMREAD_UNCHANGED)

    if type(img) is np.ndarray:

        blurred_img = cv2.blur(img, (10, 10))

        cv2.imshow('original', img)
        cv2.imshow('result', blurred_img)

        cv2.waitKey()

    print 'process end'