import glob
import os

TEST_ROOT_PATH = './results/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'input_images/cyclegan_full_color/testA/')
GT_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'input_images/cyclegan_full_color/testB/')

FULL_COLOR_CYCLEGAN_PATH = os.path.join(TEST_ROOT_PATH,
                                        'full_colors_cyclegan_all/test_latest/images')
EX_COLOR_CYCLEGAN_PATH = os.path.join(TEST_ROOT_PATH,
                                      'ex_colors_cyclegan_all/test_latest/images')
FULL_COLOR_PIX2PIX_PATH = os.path.join(TEST_ROOT_PATH,
                                       'full_colors_pix2pix_all/test_latest/images')
EX_COLOR_PIX2PIX_PATH = os.path.join(TEST_ROOT_PATH,
                                     'ex_colors_pix2pix_all/test_latest/images')
EX_COLOR_HALF_L1_PIX2PIX_PATH = os.path.join(TEST_ROOT_PATH,
                                             'ex_colors_pix2pix_all_half_l1_weight/test_latest/images')
EX_COLOR_HALF_L1_5LAYERD_PIX2PIX_PATH = os.path.join(TEST_ROOT_PATH,
                                                     'ex_colors_pix2pix_all_5layersD/test_latest/images')

test_img_list = glob.glob(ORIGINAL_IMAGE_PATH + '*.jpg')
# img_list.sort()

index = open('colorization_experiments.html', 'w')
index.write(
    "<html>\n<head>\n<title>\nColorization Test Results\n</title>\n</head>\n<body>\nColorization Test Results\n\n<table>\n")

for img_path in test_img_list:

    filename = os.path.basename(img_path)
    fake_B_name = filename[:-4] + '_fake_B.png'

    index.write("<tr>\n")
    index.write("<td>file:%s</td>\n" % (filename))

    index.write("<td><img src='%s' width=256 height=256></td>\n" % (os.path.join(ORIGINAL_IMAGE_PATH,
                                                                                 filename)))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(FULL_COLOR_CYCLEGAN_PATH,
                                                            fake_B_name)))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(EX_COLOR_CYCLEGAN_PATH,
                                                            fake_B_name)))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(FULL_COLOR_PIX2PIX_PATH,
                                                            fake_B_name)))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(EX_COLOR_PIX2PIX_PATH,
                                                            fake_B_name)))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(EX_COLOR_HALF_L1_PIX2PIX_PATH,
                                                            fake_B_name)))
    index.write("<td><img src='%s'></td>\n" % (os.path.join(EX_COLOR_HALF_L1_5LAYERD_PIX2PIX_PATH,
                                                            fake_B_name)))
    index.write("<td><img src='%s' width=256 height=256></td>\n" % (os.path.join(GT_IMAGE_PATH,
                                                                                 filename)))
    index.write("</tr>\n")

    # write captions
    index.write("<tr><th></th>\
                 <th>INPUT IMAGE</th>\
                 <th>CycleGAN (Full Color)</th>\
                 <th>CycleGAN (Uni Color)</th>\
                 <th>Pix2pix (Full Color)</th>\
                 <th>Pix2pix (Uni Color)</th>\
                 <th>Pix2pix (Uni Color. Half L1)</th>\
                 <th>Pix2pix (Uni Color. Half L1 + Global GAN)</th>\
                 <th>GT Image</th>\
                 </tr>\n")

    for idx in range(20):
        index.write("<tr></tr>\n")

index.write("</table>\n</body>\n</html>\n")

index.close()