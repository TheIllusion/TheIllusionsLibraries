import glob
import os

TEST_ROOT_PATH = '/home1/irteamsu/data/report-documents/vton-results/try-on-module/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'input')
BASELINE_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'baseline')
CUSTOM_PATH_1 = os.path.join(TEST_ROOT_PATH, 'custom1')

img_list = glob.glob(ORIGINAL_IMAGE_PATH + '/*.jpg')
img_list.sort()

index = open('vton_experiments.html', 'w')
index.write("<html>\n<body>\n<table>\n")

# write captions

for img_path in img_list:
    index.write("<tr>\n")
    index.write("<td>file:%s</td>\n" % (os.path.basename(img_path)))

    index.write("<td><img src='%s'></td>\n" % (os.path.join(ORIGINAL_IMAGE_PATH, os.path.basename(img_path))))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(BASELINE_IMAGE_PATH, os.path.basename(img_path))))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(CUSTOM_PATH_1, os.path.basename(img_path))))
    index.write("</tr>\n")

    # write captions
    index.write("<tr><th></th>\
                 <th>INPUT IMAGE</th>\
                 <th>CP-VTON(BASELINE)</th>\
                 <th>CUSTOM</th></tr>\n")


    for idx in range(20):
        index.write("<tr></tr>\n")

index.write("</table>\n</body>\n</html>\n")

index.close()