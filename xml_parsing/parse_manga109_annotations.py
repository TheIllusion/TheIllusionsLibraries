import xml.etree.ElementTree as ET
import os
import glob
import cv2

ANNOTATION_XML_DIR_ROOT = '/Users/Illusion/Documents/data/Manga109_2017_09_28/annotations/'
IMAGES_DIR_ROOT = '/Users/Illusion/Documents/data/Manga109_2017_09_28/annotations/images/'

xml_files_list = glob.glob(os.path.join(ANNOTATION_XML_DIR_ROOT, '*.xml'))

for xml_file in xml_files_list:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    main_title_name = os.path.basename(xml_file)[:-4]

    for pages in root.findall('pages'):

        for page in pages.findall('page'):

            print page.attrib.get('index')

            for frame in page.findall('frame'):
                print 'frame - ', frame.attrib

            for face in page.findall('face'):
                print 'face - ', face.attrib

            for body in page.findall('body'):
                print 'body - ', body.attrib

    print main_title_name

#tree = ET.parse('/Users/Illusion/PycharmProjects/TheIllusionsLibraries/xml_parsing/country_data.xml')
tree = ET.parse('country_data.xml')

root = tree.getroot()

for country in root.findall('country'):
    print country.attrib

    for neighbor in country.findall('neighbor'):
        print neighbor.attrib

print 'process end'