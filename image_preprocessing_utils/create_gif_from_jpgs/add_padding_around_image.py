import cv2
import numpy as np

TEST_IMAGE = '/Users/Illusion/Temp/sample.jpg'

inputImage = cv2.imread(TEST_IMAGE, cv2.IMREAD_UNCHANGED)

if type(inputImage) is not np.ndarray:
    exit(-1)

topBorderWidth = 20
bottomBorderWidth = 20
leftBorderWidth = 20
rightBorderWidth = 20
color_of_border = (255, 255, 255)

outputImage = cv2.copyMakeBorder(
                 inputImage,
                 topBorderWidth,
                 bottomBorderWidth,
                 leftBorderWidth,
                 rightBorderWidth,
                 cv2.BORDER_CONSTANT,
                 value=color_of_border
              )

cv2.imshow("output", outputImage)

cv2.waitKey()
