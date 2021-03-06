from skimage.measure import compare_ssim
import numpy as np
import math
import cv2
import argparse
import glob, os

MAX_PIXEL_VALUE = 255.0

# PSNR CALCULATION FUNC
def get_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100

    psnr = 20 * math.log10(MAX_PIXEL_VALUE / math.sqrt(mse))
    return psnr

# SSIM CALCULATION FUNC
def get_ssim(img1, img2):
    ssim = compare_ssim(img1, img2, multichannel=True)
    return ssim

# SINGLE ENCODING TRIAL
def try_encode_a_jpg(input_img, jpeg_encoding_quality, output_filename):

    cv2.imwrite(output_filename, input_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_encoding_quality])
    decoded_img = cv2.imread(output_filename, cv2.IMREAD_UNCHANGED)

    psnr_val = get_psnr(input_img, decoded_img)
    ssim_val = get_ssim(input_img, decoded_img)
    return psnr_val, ssim_val

# FIND THE BEST MATCHING ENCODING FACTOR (PSNR)
def encode_jpg_image_at_target_psnr(input_img, target_quality, output_jpg_filename):

    # dict: (quality_factor, psnr_diff)
    psnr_diff_dict = {}

    # initial max/min jpeg encoding quality (between 0,100)
    max_quality_trial_factor = 100
    min_quality_trial_factor = 0

    # initial encoding quality (the best quality)
    max_psnr_val, _ = try_encode_a_jpg(input_img, max_quality_trial_factor, "temp.jpg")
    #print 'possible max_psnr_val =', max_psnr_val

    # initial encoding quality (the lowest quality)
    min_psnr_val, _ = try_encode_a_jpg(input_img, min_quality_trial_factor, "temp.jpg")
    #print 'possible min_psnr_val =', min_psnr_val

    # error handling if the target psnr is out of the possible range
    if target_quality < min_psnr_val or target_quality > max_psnr_val:
        print 'The target psnr value cannot be achieved.'
        print 'The possible psnr range for this image is between', min_psnr_val, 'and', max_psnr_val
        return False

    # find the best matching quality factor
    while True:
        print '---------------------------------------------------------------------------'
        middle_quality_trial_factor = (max_quality_trial_factor + min_quality_trial_factor) / 2

        if middle_quality_trial_factor in psnr_diff_dict:
            break

        print 'try encoding with the jpg encoding quality factor', middle_quality_trial_factor
        current_psnr_val, current_ssim_val = try_encode_a_jpg(input_img, middle_quality_trial_factor, "temp.jpg")
        #print 'current psnr value =', current_psnr_val
        #print 'current_ssim_val =', current_ssim_val

        # record the data
        psnr_diff_dict[middle_quality_trial_factor] = abs(float(target_quality) - current_psnr_val)

        # update the next search range
        if current_psnr_val > target_quality:
            max_quality_trial_factor = middle_quality_trial_factor
        else:
            min_quality_trial_factor = middle_quality_trial_factor

    # find the minimum absolute difference between target psnr and actual psnr
    best_matching_quality_factor = min(psnr_diff_dict, key=psnr_diff_dict.get)
    print 'best matching jpg encoding quality factor =', best_matching_quality_factor

    # encode the actual jpg image
    final_psnr_val, final_ssim_val = try_encode_a_jpg(input_img, best_matching_quality_factor, output_jpg_filename)
    print 'result psnr value =', final_psnr_val
    print 'result ssim value =', final_ssim_val

    return True

# FIND THE BEST MATCHING ENCODING FACTOR (SSIM)
def encode_jpg_image_at_target_ssim(input_img, target_quality, output_jpg_filename):

    # dict: (quality_factor, ssim_diff_value)
    ssim_diff_dict = {}

    # initial max/min jpeg encoding quality (between 0,100)
    max_quality_trial_factor = 100
    min_quality_trial_factor = 0

    # initial encoding quality (the best quality)
    _, max_ssim_val = try_encode_a_jpg(input_img, max_quality_trial_factor, "temp.jpg")
    #print 'possible max_ssim_val =', max_ssim_val

    # initial encoding quality (the lowest quality)
    _, min_ssim_val = try_encode_a_jpg(input_img, min_quality_trial_factor, "temp.jpg")
    #print 'possible min_ssim_val =', min_ssim_val

    # error handling if the target psnr is out of the possible range
    if target_quality < min_ssim_val or target_quality > max_ssim_val:
        print 'The target ssim value cannot be achieved.'
        print 'The possible ssim range for this image is between', min_ssim_val, 'and', max_ssim_val
        return False

    # find the best matching quality factor
    while True:
        print '---------------------------------------------------------------------------'
        middle_quality_trial_factor = (max_quality_trial_factor + min_quality_trial_factor) / 2

        if middle_quality_trial_factor in ssim_diff_dict:
            break

        print 'try encoding with the jpg encoding quality factor', middle_quality_trial_factor
        current_psnr_val, current_ssim_val = try_encode_a_jpg(input_img, middle_quality_trial_factor, "temp.jpg")
        #print 'current psnr value =', current_psnr_val
        #print 'current_ssim_val =', current_ssim_val

        # record the data
        ssim_diff_dict[middle_quality_trial_factor] = abs(float(current_ssim_val) - target_quality)

        # update the next search range
        if current_ssim_val > target_quality:
            max_quality_trial_factor = middle_quality_trial_factor
        else:
            min_quality_trial_factor = middle_quality_trial_factor

    # find the minimum absolute difference between target psnr and actual psnr
    best_matching_quality_factor = min(ssim_diff_dict, key=ssim_diff_dict.get)
    print 'best matching jpg encoding quality factor =', best_matching_quality_factor

    # encode the actual jpg image
    final_psnr_val, final_ssim_val = try_encode_a_jpg(input_img, best_matching_quality_factor, output_jpg_filename)
    print 'result ssim value =', final_ssim_val
    print 'result psnr value =', final_psnr_val

    return True

if __name__ == '__main__':

    # single image based operation
    '''
    IS_DEBUG = True

    parser = argparse.ArgumentParser()

    if not IS_DEBUG:
        parser.add_argument('--input', type=str, help='--input inpug.png', required=True)
        parser.add_argument('--criterion', type=str, default='PSNR', help='--criterion PSNR')
        parser.add_argument('--quality', type=float, help='--quality 35', required=True)
        parser.add_argument('--output', type=str, help='--output output_encode_trial.jpg', required=True)
    else:
        # debug purposes only
        parser.add_argument('--input', type=str, default='images/jp_gates_original.png', help='--input inpug.png')
        parser.add_argument('--criterion', type=str, default='SSIM', help='--criterion PSNR')
        #parser.add_argument('--quality', type=int, default=35, help='--target 35')
        parser.add_argument('--quality', type=float, default=0.7, help='--target 0.7')
        parser.add_argument('--output', type=str, default='output_encode_trial.jpg',
                            help='--output output_encode_trial.jpg')

    args = parser.parse_args()
    print args

    input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)

    if args.criterion == 'PSNR':
        encode_jpg_image_at_target_psnr(input_img, target_quality=args.quality, output_jpg_filename=args.output)
    elif args.criterion == 'SSIM':
        encode_jpg_image_at_target_ssim(input_img, target_quality=args.quality, output_jpg_filename=args.output)
    '''

    ##########################################################################################
    # directory based operation (for mass production using psnr)
    '''
    #INPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/ava_image_image/'
    #OUTPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/encoded_jpg_images_psnr_'

    #INPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/FW_shop_original_20180220/'
    #OUTPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/FW_shop_encoded_jpg_images_psnr_'

    INPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset/'
    OUTPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset_psnr_'

    TARGET_PSNRs = [37,38,39,40,41,42,43,44,45,46,47,48,49,50]

    input_jpg_files = glob.glob(INPUT_IMAGE_DIRECTORY_PATH + '*.jpg')

    for target_psnr in TARGET_PSNRs:

        EACH_OUTPUT_IMAGE_DIRECTORY_PATH = OUTPUT_IMAGE_DIRECTORY_PATH + str(target_psnr) + '/'

        if not os.path.exists(EACH_OUTPUT_IMAGE_DIRECTORY_PATH):
            os.mkdir(EACH_OUTPUT_IMAGE_DIRECTORY_PATH)

        loop_idx = 0

        for jpg_file in input_jpg_files:

            input_img = cv2.imread(jpg_file, cv2.IMREAD_COLOR)

            ret_code = encode_jpg_image_at_target_psnr(input_img, target_quality=target_psnr, \
                                                       output_jpg_filename=EACH_OUTPUT_IMAGE_DIRECTORY_PATH + os.path.basename(jpg_file))
            print '********************************'
            print 'ret_code =', ret_code
            loop_idx = loop_idx + 1
            print 'file idx =', loop_idx
            print '********************************'
    '''
    ##########################################################################################
    # directory based operation (for mass production using ssim)

    #INPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/ava_image_image/'
    #OUTPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/encoded_jpg_images_ssim_'

    #INPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/FW_shop_original_20180220/'
    #OUTPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Downloads/FW_shop_encoded_jpg_images_ssim_'

    INPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset/'
    OUTPUT_IMAGE_DIRECTORY_PATH = '/Users/Illusion/Documents/data/shopping_mall_images/mini_testset_ssim_'

    TARGET_SSIMs = [1.0, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.90, 0.85, 0.80, 0.75]
    
    input_jpg_files = glob.glob(INPUT_IMAGE_DIRECTORY_PATH + '*.jpg')

    for target_ssim in TARGET_SSIMs:

        EACH_OUTPUT_IMAGE_DIRECTORY_PATH = OUTPUT_IMAGE_DIRECTORY_PATH + str(target_ssim) + '/'

        if not os.path.exists(EACH_OUTPUT_IMAGE_DIRECTORY_PATH):
            os.mkdir(EACH_OUTPUT_IMAGE_DIRECTORY_PATH)

        loop_idx = 0

        for jpg_file in input_jpg_files:
            input_img = cv2.imread(jpg_file, cv2.IMREAD_COLOR)

            ret_code = encode_jpg_image_at_target_ssim(input_img, target_quality=target_ssim, \
                                                       output_jpg_filename=EACH_OUTPUT_IMAGE_DIRECTORY_PATH + os.path.basename(
                                                           jpg_file))
            print '********************************'
            print 'ret_code =', ret_code
            loop_idx = loop_idx + 1
            print 'file idx =', loop_idx
            print '********************************'