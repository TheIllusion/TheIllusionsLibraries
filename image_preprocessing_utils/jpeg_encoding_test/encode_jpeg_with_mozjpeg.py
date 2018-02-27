import os, glob, time

QUALITY_FACTOR = 85

#INPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset/'
#OUTPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset_mozjpeg_q' + str(QUALITY_FACTOR) + '/'

# macbook pro
#INPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/shopping_mall/psnr_ssim_test_previous/FW_shop_original_20180220/'
#OUTPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/shopping_mall/psnr_ssim_test_previous/FW_shop_mozjpeg_q' + str(QUALITY_FACTOR) + '/'

# multiple directories (for mass processing)
# macbook pro
INPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/shopping_mall/product_c/'
original_jpg_dir_lists = glob.glob(INPUT_IMAGE_DIRECTORY + '*')

for each_directory in original_jpg_dir_lists:

    OUTPUT_IMAGE_DIRECTORY = INPUT_IMAGE_DIRECTORY + os.path.basename(each_directory) + '_mozjpeg_' + str(QUALITY_FACTOR) + '/'

    LOG_FILE = OUTPUT_IMAGE_DIRECTORY + 'q' + str(QUALITY_FACTOR) + '_process_log.txt'

    if not os.path.exists(OUTPUT_IMAGE_DIRECTORY):
        os.mkdir(OUTPUT_IMAGE_DIRECTORY)

    if not os.path.exists(each_directory):
        exit(0)
    else:
        os.chdir(each_directory)

    log_file = open(LOG_FILE, "w")
    log_file.write("filename process_time\n")

    jpg_files = glob.glob('*.jpg')

    for jpg_file in jpg_files:
        command_str = "cjpeg -quality " + str(QUALITY_FACTOR) + " -outfile " + \
                       OUTPUT_IMAGE_DIRECTORY + "output_q" + str(QUALITY_FACTOR) + '_' + jpg_file + ' ' + \
                       jpg_file

        print command_str

        start_time = time.time()

        os.system(command_str)

        elapsed_time = time.time() - start_time

        log_file.write(jpg_file + " " + str(elapsed_time) + '\n')

        print jpg_file + ' finished. elapsed_time =', str(elapsed_time) + '\n'

    log_file.close()
