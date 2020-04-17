import cv2
import numpy as np
import glob, os
#from collections import Counter

img_dir = '/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/color_split'
dest_dir = '/Volumes/Ext_850Ev/Comico/nico_ygjm_split_by_mh/64_colors_rk'

#img_path = '/Library/WebServer/Documents/LocalWebServerDoc/val_set/color_split/nicoic-02-021-027-03-f-whk-160417.jpg_00.jpg'
#img_path = '/Library/WebServer/Documents/LocalWebServerDoc/val_set/color_split/nicoic-02-021-027-03-f-whk-160417.jpg_01.jpg'
#img_path = '/Users/Illusion/Temp/rk_face.jpg'
#img_path = '/Users/Illusion/Downloads/2019-07-22 15_28_11.755.png'

#img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

def quantize_img(img):
    result_img = (img.astype(np.uint16) + 32) / 64 * 64;

    max_val = result_img > 255;
    result_img[max_val] = 255

    result_img = result_img.astype(np.uint8)

    return result_img


if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

file_list = glob.glob(os.path.join(img_dir, "*.jpg"))

idx = 0

for f in file_list:
    img = cv2.imread(f, cv2.IMREAD_COLOR)

    if (type(img) is not np.ndarray):
        print f + ' load failed!'
        continue

    result_img = quantize_img(img)

    if not (type(result_img) is not np.ndarray):
        cv2.imwrite(os.path.join(dest_dir, os.path.basename(f)[0:-4] + ".png"), result_img)

    idx += 1
    if idx % 500 == 0:
        print 'idx =', idx


'''
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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


white_pixels_idx = (img_hsv[..., 0] < 20) & (img_hsv[..., 1] < 20) & (img_hsv[..., 2] > 250)
black_pixels_idx = (img_hsv[..., 2] < 80)

#img_hsv[..., 0] = 200
#img_hsv[..., 1] = 255
#img_hsv[..., 2] = 160

# quantize img
img_hsv = img_hsv / 30 * 30

result_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

result_img[white_pixels_idx] = [255, 255, 255]
result_img[black_pixels_idx] = [0, 0, 0]

cv2.imshow('result', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
