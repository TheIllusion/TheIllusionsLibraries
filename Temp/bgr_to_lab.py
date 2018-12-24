import cv2

img = cv2.imread("/Users/Illusion/Documents/rk_face.jpg")
print img.shape

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
print img.shape

img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
print img.shape

cv2.imwrite("/Users/Illusion/Documents/out_rk_face.jpg", img)

#print img