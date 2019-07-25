import cv2
#from collections import Counter

img_path = '/Library/WebServer/Documents/LocalWebServerDoc/val_set/color_split/nicoic-02-021-027-03-f-whk-160417.jpg_00.jpg'
#img_path = '/Users/Illusion/Temp/rk_face.jpg'

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

'''
# color info in hsv space
hue = img_hsv[..., 0].flatten().tolist()
saturation = img_hsv[..., 1].flatten().tolist()
value = img_hsv[..., 2].flatten().tolist()

print 'max(hue) =', max(hue)
print 'min(hue) =', min(hue)
print 'avg(hue) =', sum(hue)/len(hue)

print 'max(saturation) =', max(saturation)
print 'min(saturation) =', min(saturation)
print 'avg(saturation) =', sum(saturation)/len(saturation)

print 'max(value) =', max(value)
print 'min(value) =', min(value)
print 'avg(value) =', sum(value)/len(value)
'''

white_pixels_idx = (img_hsv[..., 0] < 20) & (img_hsv[..., 1] < 20) & (img_hsv[..., 2] > 250)
black_pixels_idx = (img_hsv[..., 2] < 80)

#img_hsv[..., 0] = 200
img_hsv[..., 1] = 255
img_hsv[..., 2] = 160

# quantize hue
hue = img_hsv[..., 0]
img_hsv[..., 0] = hue / 10 * 10

result_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

result_img[white_pixels_idx] = [255, 255, 255]
result_img[black_pixels_idx] = [0, 0, 0]

count_list = []

cv2.imshow('result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
