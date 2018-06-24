# - *- coding: utf- 8 - *-

import cv2
import argparse
import os

# test purposes only
video_file = '/Users/Illusion/Movies/유재석_연예대상.mp4'
output_directory = '/Users/Illusion/Movies/유재석_연예대상/'

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

        if ret == True:
            output_filename = os.path.join(output_directory, str(frame_cnt) + '.jpg')
            cv2.imwrite(output_filename, frame)
        else:
            break

        frame_cnt += 1

        if frame_cnt % 100 == 0:
            print 'frame_cnt:', frame_cnt
            print 'approx_time:', float(frame_cnt)/30

    cap.release()
    #cv2.destroyAllWindows()

