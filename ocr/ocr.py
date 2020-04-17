import cv2
import pytesseract
import os, glob

# print 'tesseract methods: ', dir(pytesseract)

#IMG_PATH = '/Volumes/Ext_850Ev/OCR/doc_imgs_from_google'
IMG_PATH = '/Volumes/Ext_850Ev/Parking_Cloud/car_plate_imgs'
#IMG_PATH = '/Volumes/Ext_850Ev/OCR/ocr_test_imgs'

img_list = glob.glob(os.path.join(IMG_PATH, '*.*'))
img_list.sort()

for file in img_list:

    img = cv2.imread(file)

    print '-------------------------------------------'
    print 'Filename: ', os.path.basename(file)
    #print pytesseract.image_to_string(img)

    # Adding custom options
    #custom_config = r'--oem 2 --psm 7 -l kor'
    custom_config = r'-l kor'
    print pytesseract.image_to_string(img, config=custom_config)

