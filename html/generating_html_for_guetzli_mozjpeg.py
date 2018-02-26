import glob
import os

TEST_ROOT_PATH = '/data/report-documents/20180226-jpeg-reduction/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'mini_testset')

# guetzli
guetzli_q85_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q85')
guetzli_q90_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q90')
guetzli_q95_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q95')
guetzli_q100_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_guetzli/mini_testset_guetzli_q100')

# mozjpeg
moz_q85_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q85')
moz_q90_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q90')
moz_q95_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q95')
moz_q100_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q100')

# libjpeg (opencv)
libjpeg_psnr40_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_psnr_ssim/mini_testset_psnr_40')
libjpeg_psnr42_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_psnr_ssim/mini_testset_psnr_42')
libjpeg_psnr44_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_psnr_ssim/mini_testset_psnr_44')
libjpeg_psnr46_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_psnr_ssim/mini_testset_psnr_46')

# filelist
img_list = glob.glob(ORIGINAL_IMAGE_PATH + '/*.jpg')
img_list.sort()

index = open('result_guetzli.html', 'w')
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
    index.write("<td><img src='%s'></td>\n" % (guetzli_q85_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q90_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q95_path+'/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (guetzli_q100_path+'/' + os.path.basename(img_path)))
    index.write("</tr>\n")

    # write captions
    original_size = get_file_size(ORIGINAL_IMAGE_PATH+'/' + os.path.basename(img_path))
    guetzli_q85_size = get_file_size(guetzli_q85_path+'/' + os.path.basename(img_path))
    guetzli_q90_size = get_file_size(guetzli_q90_path + '/' + os.path.basename(img_path))
    guetzli_q95_size = get_file_size(guetzli_q95_path + '/' + os.path.basename(img_path))
    guetzli_q100_size = get_file_size(guetzli_q100_path + '/' + os.path.basename(img_path))

    index.write("<tr><th></th>\
                 <th>Original (filesize: %skB)</th>\
                 <th>Guetzli: quality-85 (filesize: %skB)</th>\
                 <th>Guetzli: quality-90 (filesize: %skB)</th>\
                 <th>Guetzli: quality-95 (filesize: %skB)</th>\
                 <th>Guetzli: quality-100 (filesize: %skB)</th></tr>\n" %
                 (original_size, guetzli_q85_size, guetzli_q90_size, guetzli_q95_size, guetzli_q100_size))

    for idx in range(30):
        index.write("<tr></tr>")
    ######################################################################################################
    # mozjpeg
    index.write("<tr>\n")
    index.write("<td></td>\n")

    index.write("<td></td>\n")
    index.write("<td><img src='%s'></td>\n" % (moz_q85_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q90_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q95_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (moz_q100_path + '/' + os.path.basename(img_path)))
    index.write("</tr>\n")

    # write captions
    original_size = get_file_size(ORIGINAL_IMAGE_PATH + '/' + os.path.basename(img_path))
    moz_q85_size = get_file_size(moz_q85_path + '/' + os.path.basename(img_path))
    moz_q90_size = get_file_size(moz_q90_path + '/' + os.path.basename(img_path))
    moz_q95_size = get_file_size(moz_q95_path + '/' + os.path.basename(img_path))
    moz_q100_size = get_file_size(moz_q100_path + '/' + os.path.basename(img_path))

    index.write("<tr><th></th>\
                     <th></th>\
                     <th>Mozjpeg: quality-85 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-90 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-95 (filesize: %skB)</th>\
                     <th>Mozjpeg: quality-100 (filesize: %skB)</th></tr>\n" %
                (moz_q85_size, moz_q90_size, moz_q95_size, moz_q100_size))

    for idx in range(30):
        index.write("<tr></tr>")
    ######################################################################################################

    ######################################################################################################
    # libjpeg (opencv)
    index.write("<tr>\n")
    index.write("<td></td>\n")

    index.write("<td></td>\n")
    index.write("<td><img src='%s'></td>\n" % (libjpeg_psnr40_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (libjpeg_psnr42_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (libjpeg_psnr44_path + '/' + os.path.basename(img_path)))
    index.write("<td><img src='%s'></td>\n" % (libjpeg_psnr46_path + '/' + os.path.basename(img_path)))
    index.write("</tr>\n")

    # write captions
    original_size = get_file_size(ORIGINAL_IMAGE_PATH + '/' + os.path.basename(img_path))
    libjpeg_psnr40_size = get_file_size(libjpeg_psnr40_path + '/' + os.path.basename(img_path))
    libjpeg_psnr42_size = get_file_size(libjpeg_psnr42_path + '/' + os.path.basename(img_path))
    libjpeg_psnr44_size = get_file_size(libjpeg_psnr44_path + '/' + os.path.basename(img_path))
    libjpeg_psnr46_size = get_file_size(libjpeg_psnr46_path + '/' + os.path.basename(img_path))

    index.write("<tr><th></th>\
                     <th></th>\
                     <th>OpenCV(libjpeg): psnr-40 (filesize: %skB)</th>\
                     <th>OpenCV(libjpeg): psnr-42 (filesize: %skB)</th>\
                     <th>OpenCV(libjpeg): psnr-44 (filesize: %skB)</th>\
                     <th>OpenCV(libjpeg): psnr-46 (filesize: %skB)</th></tr>\n" %
                (libjpeg_psnr40_size, libjpeg_psnr42_size, libjpeg_psnr44_size, libjpeg_psnr46_size))

    for idx in range(30):
        index.write("<tr></tr>")
    ######################################################################################################

index.write("</table>\n</body>\n</html>\n")
index.write("</body>\n</html>\n")

index.close()