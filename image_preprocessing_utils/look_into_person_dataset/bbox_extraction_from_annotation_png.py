import cv2
import numpy as np
import glob

lip_annotation_pngs = '/Users/Illusion/Temp/lip_annotation/*.png'

lib_annotation_filelist = glob.glob(lip_annotation_pngs)

TEST_IMG_AMOUNT = 10

# famous color codes
black = [0, 0, 0]
red = [0, 0, 255]
maroon = [0, 0, 128]
yellow = [0, 255, 255]
olive = [0, 128, 128]
lime = [0, 255, 0]
green = [0, 128, 0]
aqua = [255, 255, 0]
teal = [128, 128, 0]
blue = [255, 0, 0]
navy = [128, 0, 0]
fuchsia = [255, 0, 255]
purple = [128, 0, 128]
indianred = [92, 92, 205]
lightcoral = [128, 128, 240]
salmon = [114, 128, 250]
darksalmon = [122, 150, 233]
lightsalmon = [122, 160, 255]
silver = [192, 192, 192]
#gray = [128, 128, 128]
lightblue = [224, 185, 53]

color_list = [black, red, maroon, yellow, olive, lime, green, \
              aqua, teal, blue, navy, fuchsia, purple, \
              indianred, lightcoral, salmon, darksalmon, lightsalmon, \
              silver, lightblue]

def visualize_annotation_labels(annotation_img):

    display_img = np.zeros((annotation_img.shape[0], annotation_img.shape[1], 3), np.uint8)
    print 'display_img.shape =', display_img.shape

    for label_idx in range(0, 20):
        idx = (annotation_img[...] == label_idx)
        display_img[idx] = color_list[label_idx]

    return display_img

loop_idx = 0
display_img = []

for png in lib_annotation_filelist:
    img = cv2.imread(png, cv2.IMREAD_UNCHANGED)

    print 'img.shape =', img.shape

    display_img.append(visualize_annotation_labels(img))

    print 'filename =', png

    window_name_annotation = 'annotaion_' + str(loop_idx)
    window_name_display = 'annotaion_display_' + str(loop_idx)

    #cv2.imshow(window_name_annotation, img)
    cv2.imshow(window_name_display, display_img[loop_idx])
    cv2.waitKey()
    #cv2.destroyWindow(window_name_annotation)
    cv2.destroyWindow(window_name_display)

    loop_idx += 1

    if loop_idx > 10:
        break

# appendix
# the range of annotation label is from 0 to 19
# Class Definition
# 0 Background
# 1 Hat
# 2 Hair
# 3 Glove
# 4 Sunglasses
# 5 Upper-clothes
# 6 Dress
# 7 Coat
# 8 Socks
# 9 Pants
# 10 Jumpsuits
# 11 Scarf
# 12 Skirt
# 13 Face
# 14 Left-arm
# 15 Right-arm
# 16 Left-leg
# 17 Right-leg
# 18 Left-shoe
# 19 Right-shoe