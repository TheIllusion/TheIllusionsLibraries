import cv2

img = cv2.imread("/Users/Illusion/Downloads/hair_original.jpg")

width, height, ch = img.shape

resized_img = cv2.resize(img, (height*2, width*2), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("/Users/Illusion/Downloads/hair_4x_bilinear.jpg", resized_img)