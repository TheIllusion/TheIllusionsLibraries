import toastlib.facedetect as fd
import toastlib.imageutils as iu
import numpy as np
import cv2
import os
import glob
import copy

# load face detector
face_det = fd.get_detector_dlib()

def face_detection( frame ):

    # recognition buffer
    frame_r   = copy.deepcopy( frame )

    # face detection
    faces = fd.detect_dlib( face_det, frame_r )

    # can't detect
    if len( faces ) < 1:
        return [], []

    # for each detected faces
    ( x, y, w, h ) = faces[0]
    pos = [ x, y, w, h ]

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        print 'weired pos=', pos
        return [], []

    # extract face image
    '''
    if w > h:
        face_image = frame_r[ y:y+h, x+w/2-h/2:x+w/2+h/2 ]
    elif w < h:
        face_image = frame_r[ y+h/2-w/2:y+h/2+w/2, x:x+w ]
    else:
        face_image = frame_r[ y:y+h, x:x+w ]
    '''

    if y - h/3 > 0:
        y_start = y - h/3
    else:
        y_start = 0

    if y + h / 5 > frame.shape[0]:
        y_end = frame.shape[0] - 1
    else:
        y_end = y + h+ h / 5

    if x - w / 7 > 0:
        x_start = x - w / 7
    else:
        x_start = 0

    if x + w / 7 > frame.shape[1]:
        x_end = frame.shape[1] - 1
    else:
        x_end = x + w + w / 7

    face_image = frame_r[ y_start : y_end, x_start : x_end ]

    return face_image, pos

if __name__ == "__main__":

    jpg_files = glob.glob( '*.jpg' )

    for jpg_file in jpg_files:

        # load image (exif_orientation = false when opencv-3)
        try:
            frame = iu.load_image(jpg_file, exif_orientation=False )
        except:
            print "Exception occurred in " + jpg_file
            continue

        if (type(frame) is not np.ndarray):
            print jpg_file + ' load failed!'
            continue

        # face detection
        try:
            face_image, pos = face_detection( frame )
        except:
            print "Exception occurred in " + jpg_file
            continue

        # can't detect a face
        if len(pos) < 1:
            print "Cant' detect face in " + jpg_file
            continue

        # save face image
        cv2.imwrite('result_' + jpg_file, face_image)
