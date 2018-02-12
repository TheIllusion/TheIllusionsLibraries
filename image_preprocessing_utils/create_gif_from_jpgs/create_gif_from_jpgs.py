import sys, os, glob
import datetime
import imageio

'''Only png and jpg files are allowed as input format'''
VALID_EXTENSIONS = ('png', 'jpg')

INPUT_IMAGE_PATH = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/image_preprocessing_utils/python-compare-two-images/images/'
RESULT_GIF_PATH = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/image_preprocessing_utils/create_gif_from_jpgs/'

def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(RESULT_GIF_PATH + output_file, images, duration=duration)


if __name__ == "__main__":

    '''original open-source form http://www.idiotinside.com/2017/06/06/create-gif-animation-with-python/'''
    '''
    script = sys.argv.pop(0)

    if len(sys.argv) < 2:
        print('Usage: python {} <duration> <path to images separated by space>'.format(script))
        sys.exit(1)

    duration = float(sys.argv.pop(0))
    filenames = sys.argv

    if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
        print('Only png and jpg files allowed')
        sys.exit(1)

    create_gif(filenames, duration)
    '''

    '''code written by rklee'''

    print 'hi'

    duration = 5

    filenames = glob.glob(INPUT_IMAGE_PATH + '*.png')

    if len(filenames) > 1:
        create_gif(filenames, duration)
    else:
        print 'empty filelist'