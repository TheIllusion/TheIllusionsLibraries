import glob
import os

#TEST_ROOT_PATH = '/Users/Illusion/Documents/Data/shopping_mall/'
TEST_ROOT_PATH = '/data/report-documents/20180226-jpeg-reduction/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'mini_testset')

# mozjpeg
moz_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_samesize_libjpeg_psnr37')

# libjpeg (opencv)
libjpeg_psnr37_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_psnr_ssim/mini_testset_psnr_37')

# filelist
img_list = glob.glob(moz_path + '/*.jpg')
img_list.sort()

index = open('result_mozjpeg_vs_libjpeg.html', 'w')
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

    ######################################################################################################
    # guetzli
    index.write("<tr>\n")
    index.write("<td>file:%s</td>\n" % (os.path.basename(img_path)))

    index.write("<td><img src='%s'></td>\n" % (ORIGINAL_IMAGE_PATH+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (libjpeg_psnr37_path+'/' + os.path.basename(img_path)))
    index.write("</tr>\n")

    # write captions
    original_size = get_file_size(ORIGINAL_IMAGE_PATH+'/' + os.path.basename(img_path))
    moz_size = get_file_size(moz_path+'/' + os.path.basename(img_path))
    libjpeg_size = get_file_size(libjpeg_psnr37_path + '/' + os.path.basename(img_path))

    index.write("<tr><th></th>\
                 <th>Original (filesize: %skB)</th>\
                 <th>Mozjpeg (filesize: %skB)</th>\
                 <th>libjpeg (filesize: %skB)</th>\n" %\
                 (original_size, moz_size, libjpeg_size))

    for idx in range(50):
        index.write("<tr></tr>")

    ######################################################################################################

index.write("</table>\n</body>\n</html>\n")
index.write("</body>\n</html>\n")

index.close()