#from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim
import numpy as np
import math
import cv2

MAX_PIXEL_VALUE = 255.0

'''
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
'''

def get_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100

    psnr = 20 * math.log10(MAX_PIXEL_VALUE / math.sqrt(mse))
    return psnr

def get_ssim(img1, img2):
    ssim = compare_ssim(img1, img2, multichannel=True)
    return ssim

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    p = get_psnr(imageA, imageB)
    s = get_ssim(imageA, imageB)

    print 'psnr =', p
    print 'ssime =', s

def encode_jpg_image_at_target_psnr(input_img, target_psnr):

    # initial encoding quality
    jpeg_encoding_quality = 100
    cv2.imwrite('output_' + str(target_psnr) + '.jpg', input_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_encoding_quality])
    decoded_img = cv2.imread('output_' + str(target_psnr) + '.jpg', cv2.IMREAD_UNCHANGED)
    max_psnr_val = get_psnr(input_img, decoded_img)
    print 'max_psnr_val =', max_psnr_val

    jpeg_encoding_quality = 0
    cv2.imwrite('output_' + str(target_psnr) + '.jpg', input_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_encoding_quality])
    decoded_img = cv2.imread('output_' + str(target_psnr) + '.jpg', cv2.IMREAD_UNCHANGED)
    min_psnr_val = get_psnr(input_img, decoded_img)
    print 'min_psnr_val =', min_psnr_val




    print psnr_val

# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("images/jp_gates_original.png")
original_grayscale = cv2.imread("images/jp_gates_original.png", cv2.IMREAD_GRAYSCALE)
contrast = cv2.imread("images/jp_gates_contrast.png")
shopped = cv2.imread("images/jp_gates_photoshopped.png")

encode_jpg_image_at_psnr(original, 50)

'''
print 'original vs original'
compare_images(original, original)

print 'original vs original(grayscale)'
compare_images(original_grayscale, original_grayscale)

print 'original vs contrast'
compare_images(original, contrast)

print 'original vs photoshopped'
compare_images(original, shopped)
'''

