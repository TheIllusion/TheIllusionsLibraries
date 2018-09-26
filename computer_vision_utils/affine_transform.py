# please refer to https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/Illusion/Pictures/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')

rows,cols,ch = img.shape

print 'img.shape =', img.shape

pts1 = np.float32([[0,0],[200,0],[0,500]])
pts2 = np.float32([[300,0],[500,0],[150,500]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(int(cols*1.3),int(rows)))

# bgr to rgb
img_rgb = img.copy()
img_rgb[:, :, 0 ] = img[:, :, 2]
img_rgb[:, :, 1 ] = img[:, :, 1]
img_rgb[:, :, 2 ] = img[:, :, 0]

dst_rgb = dst.copy()
dst_rgb[:, :, 0 ] = dst[:, :, 2]
dst_rgb[:, :, 1 ] = dst[:, :, 1]
dst_rgb[:, :, 2 ] = dst[:, :, 0]

plt.subplot(121),plt.imshow(img_rgb),plt.title('Input')
plt.subplot(122),plt.imshow(dst_rgb),plt.title('Output')
plt.show()