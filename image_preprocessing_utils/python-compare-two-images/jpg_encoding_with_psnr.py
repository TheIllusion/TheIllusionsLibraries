from skimage.measure import compare_ssim
import numpy as np
import math
import cv2
import argparse

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

# FIND THE BEST MATCHING ENCODING FACTOR
def encode_jpg_image_at_target_psnr(input_img, target_psnr_val, output_jpg_filename):

    # dict: (quality_factor, psnr_diff)
    psnr_diff_dict = {}

    # initial max/min jpeg encoding quality (between 0,100)
    max_quality_trial_factor = 100
    min_quality_trial_factor = 0

    # initial encoding quality (the best quality)
    max_psnr_val, _ = try_encode_a_jpg(input_img, max_quality_trial_factor, output_jpg_filename)
    print 'possible max_psnr_val =', max_psnr_val

    # initial encoding quality (the lowest quality)
    min_psnr_val, _ = try_encode_a_jpg(input_img, min_quality_trial_factor, output_jpg_filename)
    print 'possible min_psnr_val =', min_psnr_val

    # error handling if the target psnr is out of the possible range
    if target_psnr_val < min_psnr_val or target_psnr_val > max_psnr_val:
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
        current_psnr_val, current_ssim_val = try_encode_a_jpg(input_img, middle_quality_trial_factor, output_jpg_filename)
        print 'current psnr value =', current_psnr_val
        print 'current_ssim_val =', current_ssim_val

        # record the data
        psnr_diff_dict[middle_quality_trial_factor] = abs(float(target_psnr_val) - current_psnr_val)

        # update the next search range
        if current_psnr_val > target_psnr_val:
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='--input inpug.png', required=True)
    parser.add_argument('--psnr', type=int, help='--psnr 35', required=True)
    parser.add_argument('--output', type=str, help='--output output_encode_trial.jpg', required=True)

    # debug purposes only
    '''
    parser.add_argument('--input', type=str, default='images/jp_gates_original.png', help='--input inpug.png')
    parser.add_argument('--psnr', type=int, default=35, help='--psnr 35')
    parser.add_argument('--output', type=str, default='output_encode_trial.jpg',
                        help='--output output_encode_trial.jpg')
    '''

    args = parser.parse_args()
    print args

    input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    encode_jpg_image_at_target_psnr(input_img, target_psnr_val=args.psnr, output_jpg_filename=args.output)
