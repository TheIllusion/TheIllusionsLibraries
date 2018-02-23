import os, glob, time

QUALITY_FACTOR = 85

#INPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset/'
#OUTPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset_mozjpeg_q' + str(QUALITY_FACTOR) + '/'

# macbook pro
INPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/shopping_mall/speed_test/'
OUTPUT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/shopping_mall/speed_test_mozjpeg_q' + str(QUALITY_FACTOR) + '/'

LOG_FILE = OUTPUT_IMAGE_DIRECTORY + 'q' + str(QUALITY_FACTOR) + '_process_log.txt'

if not os.path.exists(OUTPUT_IMAGE_DIRECTORY):
    os.mkdir(OUTPUT_IMAGE_DIRECTORY)

if not os.path.exists(INPUT_IMAGE_DIRECTORY):
    exit(0)
else:
    os.chdir(INPUT_IMAGE_DIRECTORY)

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