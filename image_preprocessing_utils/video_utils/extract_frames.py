# - *- coding: utf- 8 - *-

import cv2
import argparse
import os

# test purposes only
video_file = '/Users/Illusion/Movies/손석희_인터뷰.mp4'
output_directory = '/Users/Illusion/Movies/손석희_인터뷰/'

# get args
def get_args():
    global video_file
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, help='input video filename', required=True)
    parser.add_argument('-o', '--outdir', type=str, help='output jpg directory', required=True)
    args = parser.parse_args()
    video_file = args.infile
    output_directory = args.outdir

if __name__ == '__main__':

    #get_args()

    # open video file
    cap = cv2.VideoCapture(video_file)

    frame_cnt = 0

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    while (cap.isOpened()):
        ret, frame = cap.read()

        '''
        if ret == True:
            cv2.imshow('frame', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''

        output_filename = os.path.join(output_directory, str(frame_cnt) + '.jpg')

        cv2.imwrite(output_filename, frame)

        frame_cnt += 1

        if frame_cnt % 100 == 0:
            print 'frame_cnt:', frame_cnt

    cap.release()
    #cv2.destroyAllWindows()

