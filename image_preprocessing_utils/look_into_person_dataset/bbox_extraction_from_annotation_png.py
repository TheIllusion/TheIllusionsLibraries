import cv2
import numpy as np
import glob
import os

lip_annotation_pngs = '/Users/Illusion/Documents/Data_public_set/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/*.png'
lip_img_path = '/Users/Illusion/Documents/Data_public_set/LIP/TrainVal_images/val_images/'

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

annotation_class_dict = {"Background": 0, "Hat": 1, "Hair": 2, "Glove": 3, "Sunglasses": 4, \
                         "Upper-clothes": 5, "Dress": 6, "Coat": 7, "Socks": 8, "Pants": 9, \
                         "Jumpsuits": 10, "Scarf": 11, "Skirt": 12, "Face": 13, "Left-arm": 14, \
                         "Right-arm": 15, "Left-leg": 16, "Right-leg": 17, "Left-shoe": 18, "Right-shoe": 19}


# for key in annotation_class_dict:
#     print 'key =', key, ' :  value =', annotation_class_dict[key]

def visualize_annotation_labels(annotation_img):

    display_img = np.zeros((annotation_img.shape[0], annotation_img.shape[1], 3), np.uint8)
    print 'display_img.shape =', display_img.shape

    # visualize all categories
    # for label_idx in range(0, 20):
    #     idx = (annotation_img[...] == label_idx)
    #     display_img[idx] = color_list[label_idx]

    # visualize the specific label
    idx = (annotation_img[...] == annotation_class_dict["Pants"])
    display_img[idx] = color_list[annotation_class_dict["Pants"]]

    return display_img


def find_bbox(color_img):

    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("black and white", img)

    ret, thresh = cv2.threshold(img, 5, 255, 0)
    cv2.imshow("thresh", thresh)

    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    #cv2.imshow("im2", im2)
    #cv2.waitKey()

    if(len(contours) > 0):

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # show bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        # color_img = cv2.rectangle(color_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # cv2.imshow("bounding rectangle", color_img)
        # cv2.waitKey()

        return True, x, y, w, h

    else:
        return False, 0, 0, 0, 0


loop_idx = 0
display_img = []

for png in lib_annotation_filelist:
    img = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    original_img = cv2.imread(os.path.join(lip_img_path, os.path.basename(png)[:-4] + '.jpg'))

    if (type(img) is not np.ndarray) or (type(original_img) is not np.ndarray):
        continue

    print 'img.shape =', img.shape

    result_img = visualize_annotation_labels(img)

    display_img.append(result_img)

    ret, x, y, w, h = find_bbox(result_img)

    if ret == True:
        extracted_img = original_img[y:y+h, x:x+w]
        cv2.imshow("extracted_img", extracted_img)
        cv2.waitKey()

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