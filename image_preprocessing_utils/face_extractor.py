import dlib
import os, glob
import cv2

INPUT_DIR = '/Users/Illusion/Downloads/trainB_blonde/'
OUTPUT_DIR = '/Users/Illusion/Downloads/trainB_blonde_face_cropped/'

detector = dlib.get_frontal_face_detector()

def face_detection(img):
    #     rgbImg = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    rgbImg = img[:, :, ::-1]
    # @param  upsample Upsamping factor before detection (0 = 1x, 1 = 2x, ...) [Default:0]
    upsample = 0
    dets = detector(rgbImg, upsample)

    # convert format
    results = []
    for i, d in enumerate(dets):
        results.append((d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()))

    return results

def face_margin_crop(img, face_rect):
    (x, y, w, h) = face_rect[0]

    margin = 0.5

    x_m = int(x - w * margin)
    # more margin for top part
    y_m = int(y - h * (margin + 0.2))

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

    return face_m

if __name__ == '__main__':

    os.chdir(INPUT_DIR)

    dest_path = os.path.join(OUTPUT_DIR)

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    img_files = glob.glob('*.jpg')

    img_idx = 0

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)

        face_rect = face_detection(img)

        if len(face_rect) == 0:
            continue

        try:
            face_img = face_margin_crop(img, face_rect)
        except:
            print '!Error : Face Crop Failed.'
            continue

        cv2.imwrite(os.path.join(dest_path, img_file), face_img)

        img_idx = img_idx + 1
        if img_idx % 100 == 0:
            print 'index: ', str(img_idx)
