# Manga109 dataset parser
# refer to 'https://docs.python.org/2/library/xml.etree.elementtree.html#module-xml.etree.ElementTree'

import xml.etree.ElementTree as ET
import os
import glob
import cv2
import numpy as np

# dataset
ANNOTATION_XML_DIR_ROOT = '/Users/Illusion/Documents/data/Manga109_2017_09_28/annotations/'
IMAGES_DIR_ROOT = '/Users/Illusion/Documents/data/Manga109_2017_09_28/images/'

# output directories
OUTPUT_FACES_DIR = '/Users/Illusion/Documents/data/Manga109_refined/faces/'
OUTPUT_BODIES_DIR = '/Users/Illusion/Documents/data/Manga109_refined/bodies/'
OUTPUT_FRAMES_DIR = '/Users/Illusion/Documents/data/Manga109_refined/frames/'

xml_files_list = glob.glob(os.path.join(ANNOTATION_XML_DIR_ROOT, '*.xml'))

if not os.path.exists(OUTPUT_FACES_DIR):
    os.mkdir(OUTPUT_FACES_DIR)

if not os.path.exists(OUTPUT_BODIES_DIR):
    os.mkdir(OUTPUT_BODIES_DIR)

if not os.path.exists(OUTPUT_FRAMES_DIR):
    os.mkdir(OUTPUT_FRAMES_DIR)

for xml_file in xml_files_list:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    main_title_name = os.path.basename(xml_file)[:-4]

    for pages in root.findall('pages'):

        for page in pages.findall('page'):

            index = format(int(page.attrib.get('index')), '03d')
            print index

            # open the image file
            img_filename = os.path.join(IMAGES_DIR_ROOT,
                                          main_title_name,
                                          str(index) + '.jpg')
            img = cv2.imread(img_filename)
            if type(img) is not np.ndarray:
                print 'file loading error'
                continue

            # parse frames
            # e.g. <frame id="00002cff" xmin="81" ymin="584" xmax="775" ymax="1094"/>
            for frame in page.findall('frame'):
                #print 'frame - ', frame.attrib
                id = frame.attrib.get('id')
                xmin = int(frame.attrib.get('xmin'))
                ymin = int(frame.attrib.get('ymin'))
                xmax = int(frame.attrib.get('xmax'))
                ymax = int(frame.attrib.get('ymax'))

                # crop and save img
                cropped_img = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(OUTPUT_FRAMES_DIR, id + '.jpg'), cropped_img)

            # parse faces
            # e.g. <face id="00002d13" xmin="957" ymin="975" xmax="1029" ymax="1056" character="00002d05"/>
            for face in page.findall('face'):
                #print 'face - ', face.attrib
                id = face.attrib.get('id')
                xmin = int(face.attrib.get('xmin'))
                ymin = int(face.attrib.get('ymin'))
                xmax = int(face.attrib.get('xmax'))
                ymax = int(face.attrib.get('ymax'))

                # crop and save img
                cropped_img = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(OUTPUT_FACES_DIR, id + '.jpg'), cropped_img)

            # parse bodies
            # e.g. <body id="00002d27" xmin="653" ymin="972" xmax="728" ymax="1055" character="00002d03"/>
            for body in page.findall('body'):
                #print 'body - ', body.attrib
                id = body.attrib.get('id')
                xmin = int(body.attrib.get('xmin'))
                ymin = int(body.attrib.get('ymin'))
                xmax = int(body.attrib.get('xmax'))
                ymax = int(body.attrib.get('ymax'))

                # crop and save img
                cropped_img = img[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(OUTPUT_BODIES_DIR, id + '.jpg'), cropped_img)

    print main_title_name

#tree = ET.parse('/Users/Illusion/PycharmProjects/TheIllusionsLibraries/xml_parsing/country_data.xml')
tree = ET.parse('country_data.xml')

root = tree.getroot()

for country in root.findall('country'):
    print country.attrib

    for neighbor in country.findall('neighbor'):
        print neighbor.attrib

print 'process end'