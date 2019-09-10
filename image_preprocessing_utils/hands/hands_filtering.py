import cv2
import numpy as np

HAND_IMG1 = '/Users/Illusion/Temp/rk.jpg'

if __name__ == '__main__':
    img = cv2.imread(HAND_IMG1, cv2.IMREAD_UNCHANGED)

    if type(img) is np.ndarray:

        result = cv2.bilateralFilter(img, d=30, sigmaColor=100, sigmaSpace=100)

        concated_img = np.hstack((img, result))

        cv2.imshow('result', concated_img)

        cv2.waitKey()

    print 'process end'