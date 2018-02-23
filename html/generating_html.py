import glob
import os

TEST_ROOT_PATH = '/Users/Illusion/Documents/Data/shopping_mall/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'mini_testset')
guetzli_q85_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q85')
guetzli_q90_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q90')
guetzli_q95_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q95')
guetzli_q100_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q100')

img_list = glob.glob(ORIGINAL_IMAGE_PATH + '/*.jpg')
img_list.sort()

index = open('result_guetzli.html', 'w')
index.write("<html>\n<body>\n<table>\n")

# write captions

for img_path in img_list:
    index.write("<tr>\n")
    index.write("<td>file:%s</td>\n" % (os.path.basename(img_path)))

    index.write("<td><img src='%s'></td>\n" % (ORIGINAL_IMAGE_PATH+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q85_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q90_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q95_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q100_path+'/' + os.path.basename(img_path)))
    index.write("</tr>\n")

    # write captions
    index.write("<tr><th></th>\
                 <th>Original Image</th>\
                 <th>Guetzli-Quality-85</th>\
                 <th>Guetzli-Quality-90</th>\
                 <th>Guetzli-Quality-95</th>\
                 <th>Guetzli-Quality-100</th></tr>\n")


    for idx in range(20):
        index.write("<tr></tr>\n")

index.write("</table>\n</body>\n</html>\n")

index.close()