from PIL import Image
from PIL import ImageEnhance

import glob
import os

os.chdir("/Users/Illusion/Desktop/temp/faces_ori")

image_files = glob.glob('*.png')

for filename in image_files:

    image = Image.open(filename)

    #enhancer_sharpness = ImageEnhance.Sharpness(image)

    #enhancer_contrast = ImageEnhance.Contrast(enhancer_sharpness.enhance(1.1))

    enhancer_contrast = ImageEnhance.Contrast(image)

    enhancer_color = ImageEnhance.Color(enhancer_contrast.enhance(2))

    enhancer_color.enhance(0.7).save("result_" + filename)
