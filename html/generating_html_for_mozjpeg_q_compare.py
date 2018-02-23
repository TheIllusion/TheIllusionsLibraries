import glob
import os

TEST_ROOT_PATH = '/Users/Illusion/Documents/Data/shopping_mall/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'mini_testset')

# mozjpeg
moz_q50_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q50')
moz_q55_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q55')
moz_q60_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q60')
moz_q65_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q65')
moz_q70_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q70')
moz_q75_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q75')
moz_q80_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q80')
moz_q85_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q85')
moz_q90_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q90')
moz_q95_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q95')
moz_q100_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q100')

# filelist
img_list = glob.glob(ORIGINAL_IMAGE_PATH + '/*.jpg')
img_list.sort()

index = open('result_mozjpeg_q_compare.html', 'w')
index.write("<html>\n<body>\n<table>\n")

def get_file_size(full_path):
    try:
        size = os.path.getsize(full_path)
        if size != 0:
            size = size / 1000
        return size
    except:
        return 0

index.write("<html>\n<body>\n")

for img_path in img_list:

    # mozjpeg
    index.write("<tr>\n")

    index.write("<td>file:%s</td>\n" % (os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (ORIGINAL_IMAGE_PATH + '/' + os.path.basename(img_path)))

    index.write("<td><img src='%s'></td>\n" % (moz_q50_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q55_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q60_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q65_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q70_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q75_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q80_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q85_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q90_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q95_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q100_path + '/' + os.path.basename(img_path)))
    index.write("</tr>\n")

    # write captions
    original_size = get_file_size(ORIGINAL_IMAGE_PATH + '/' + os.path.basename(img_path))
    moz_q50_size = get_file_size(moz_q50_path + '/' + os.path.basename(img_path))
    moz_q55_size = get_file_size(moz_q55_path + '/' + os.path.basename(img_path))
    moz_q60_size = get_file_size(moz_q60_path + '/' + os.path.basename(img_path))
    moz_q65_size = get_file_size(moz_q65_path + '/' + os.path.basename(img_path))
    moz_q70_size = get_file_size(moz_q70_path + '/' + os.path.basename(img_path))
    moz_q75_size = get_file_size(moz_q75_path + '/' + os.path.basename(img_path))
    moz_q80_size = get_file_size(moz_q80_path + '/' + os.path.basename(img_path))
    moz_q85_size = get_file_size(moz_q85_path + '/' + os.path.basename(img_path))
    moz_q90_size = get_file_size(moz_q90_path + '/' + os.path.basename(img_path))
    moz_q95_size = get_file_size(moz_q95_path + '/' + os.path.basename(img_path))
    moz_q100_size = get_file_size(moz_q100_path + '/' + os.path.basename(img_path))

    index.write("<tr><th></th>\
                     <th>Original (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-50 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-55 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-60 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-65 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-70 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-75 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-80 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-85 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-90 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-95 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-100 (filesize: %skB)</th></tr>\n" %
                (original_size, moz_q50_size, moz_q55_size, moz_q60_size, moz_q65_size, moz_q70_size, \
                 moz_q75_size, moz_q80_size, moz_q85_size, moz_q90_size, moz_q95_size, moz_q100_size))

    for idx in range(30):
        index.write("<tr></tr>")
    ######################################################################################################

index.write("</table>\n</body>\n</html>\n")
index.write("</body>\n</html>\n")

index.close()