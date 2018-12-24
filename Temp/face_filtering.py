import toastlib.facedetect as fd
import cv2
import glob
import os
import time
import os
import sys

walk_dir = './images/hair'

def crop_face_img(img_path, dest_path):
    try:
        img = cv2.imread(img_path)
        face_det = fd.get_detector_dlib()
        faces = fd.detect_dlib(face_det, img)

        if len(faces) > 0 and len(img) > 0 and len(img[0]) > 0:
            (x, y, w, h) = faces[0]
            margin = 0.5

            x_m = int(x - w * margin)
            # more margin for top part
            y_m = int(y - h * 0.7)

            if (x_m < 0):
                x_m = 0
            if (y_m < 0):
                y_m = 0

            x_rect_m = int(x + w + w * margin)
            y_rect_m = int(y + h + h * margin)

            if (x_rect_m > len(img[0])):
                x_rect_m = len(img[0])

            if (y_rect_m > len(img)):
                y_rect_m = len(img)

            face_m = img[y_m:y_rect_m, x_m:x_rect_m, :]
            directory = os.path.dirname(dest_path)
            if not os.path.exists(directory):
                    os.makedirs(directory)
            cv2.imwrite(dest_path, face_m)
            print(os.path.basename(img_path))
    except:
        print('exception')

if __name__ == '__main__':
    # hashtag_2_process = 'chinese'
    #
    # img_list = glob.glob('./' + hashtag_2_process + '/*.jpg')
    # img_list.sort()
    #
    # count = 0
    print('walk_dir = ' + walk_dir)

    # If your current working directory may change during script execution, it's recommended to
    # immediately convert program arguments to an absolute path. Then the variable root below will
    # be an absolute path as well. Example:
    # walk_dir = os.path.abspath(walk_dir)
    print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))

    for root, subdirs, files in os.walk(walk_dir):
        print('--\nroot = ' + root)
        list_file_path = os.path.join(root, 'my-directory-list.txt')
        print('list_file_path = ' + list_file_path)

        with open(list_file_path, 'wb') as list_file:
            for subdir in subdirs:
                print('\t- subdirectory ' + subdir)

            for filename in files:
                file_path = os.path.join(root, filename)
                print('\t- file %s (full path: %s)' % (filename, file_path))
                dest_path = './face_crop/' + file_path.lstrip('./')
                print('\t - destination path : %s' % (dest_path))
                crop_face_img(file_path, dest_path)





