# Manga109 dataset parser
# refer to 'https://docs.python.org/2/library/xml.etree.elementtree.html#module-xml.etree.ElementTree'

import xml.etree.ElementTree as ET
import os
import glob
import cv2
import numpy as np

# dataset
ANNOTATION_XML_DIR_ROOT = '/Users/Illusion/Documents/data/stanford_dog_dataset/Annotation/'
IMAGES_DIR_ROOT = '/Users/Illusion/Documents/data/stanford_dog_dataset/images/'

# output directories
EXTRACTED_OUTPUT_DIR = '/Users/Illusion/Documents/data/stanford_dog_dataset/extracted_results/'

if not os.path.exists(EXTRACTED_OUTPUT_DIR):
    os.mkdir(EXTRACTED_OUTPUT_DIR)

os.chdir(ANNOTATION_XML_DIR_ROOT)
dir_list = glob.glob("*")

processed_img_count = 0

for dir in dir_list:
    os.chdir(os.path.join(ANNOTATION_XML_DIR_ROOT, dir))
    xml_files_list = glob.glob('*')

    for xml_file in xml_files_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        obj_count = 0

        for object in root.findall('object'):

            # open the image file
            img_filename = os.path.join(IMAGES_DIR_ROOT,
                                          dir,
                                          xml_file + '.jpg')

            img = cv2.imread(img_filename)
            if type(img) is not np.ndarray:
                print 'file loading error'
                continue

            # parse bbox
            for bbox in object.findall('bndbox'):
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # crop and save img
                cropped_img = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(EXTRACTED_OUTPUT_DIR, xml_file + '_obj_' + str(obj_count) + '.jpg'), cropped_img)

                obj_count += 1

        processed_img_count += 1

        if processed_img_count % 100 == 0:
            print 'processed_img_count =', processed_img_count

print 'process end'