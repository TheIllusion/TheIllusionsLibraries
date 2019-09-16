import cv2
import numpy as np
import time

#HAND_IMG1 = '/Users/Illusion/Temp/rk.jpg'
ANI_IMG1 = '/Users/Illusion/Temp/epoch166_fake_B.png'


if __name__ == '__main__':
    img = cv2.imread(ANI_IMG1, cv2.IMREAD_UNCHANGED)

    if type(img) is np.ndarray:

        start_time = time.time()

        result = cv2.bilateralFilter(img, d=50, sigmaColor=90, sigmaSpace=80)

        elapsed_time = time.time() - start_time

        concated_img = np.hstack((img, result))

        cv2.imshow('result', concated_img)

        print 'elapsed_time: ', elapsed_time

        cv2.waitKey()

    print 'process end'