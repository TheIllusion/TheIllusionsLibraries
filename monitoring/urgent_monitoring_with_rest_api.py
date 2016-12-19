import time
import os
import urllib2
import datetime
import re

REQUEST_SENDING_INTERVAL = 30
REQUEST_TIMEOUT_THRESHOLD_SEC = 30
RESPONSE_TIME_WARINING_THRESHOLD = 5

FILE_PATH_FOR_ALL_LOG = '/Users/Illusion/Temp/response_all_log.txt'
FILE_PATH_FOR_WARNING_LOG = '/Users/Illusion/Temp/response_warning.log'

#rest_api_address = 'http://vincent.nhnent.com:8979/hand'
#hand_rest_api_address = 'http://10.165.128.51:8979/hand'

hand_rest_api_address = []
face_rest_api_address = []

hand_rest_api_address.append('http://10.161.31.22:8979/hand')
hand_rest_api_address.append('http://10.161.31.23:8979/hand')
hand_rest_api_address.append('http://10.161.31.24:8979/hand')
hand_rest_api_address.append('http://10.161.31.25:8979/hand')

face_rest_api_address.append('http://10.161.31.22:8989/face')
face_rest_api_address.append('http://10.161.31.23:8989/face')
face_rest_api_address.append('http://10.161.31.24:8989/face')
face_rest_api_address.append('http://10.161.31.25:8989/face')

#hand_img_path = '/Users/Illusion/Documents/Palm_Data/random_hands/IMG_0703.jpg'

#12' Macbook
#hand_img_path = '/Users/Illusion/Documents/rk.jpg'
#face_img_path = '/Users/Illusion/Pictures/sulhwa.png'

#Macbook Pro
hand_img_path = '/Users/Illusion/Documents/Data/palm_data/test_set/crop_resize_512_512/rk.jpg'
face_img_path = '/Users/Illusion/Documents/Data/toast_faces/rk.jpg'

f_hand = open(hand_img_path, 'rb')
hand_image = f_hand.read()
f_hand.close()

f_face = open(face_img_path, 'rb')
face_image = f_face.read()
f_face.close()

index = 0

while True:

    ###########################################################################
    # Check response time for hand recognition

    print '==================================================================='
    print 'hand ip = ', hand_rest_api_address[index]
    req = urllib2.Request(hand_rest_api_address[index], data = hand_image)
    req.add_header('Content-Length', '%d' % os.path.getsize(hand_img_path))
    req.add_header('Content-Type', 'application/octet-stream')

    start_time = time.time()

    res = urllib2.urlopen(req, timeout=REQUEST_TIMEOUT_THRESHOLD_SEC).read().strip()

    elapsed_time = time.time() - start_time

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    line_string = current_time + ' Elapsed time(hand) = ' + str(elapsed_time) + '\n'
    print line_string

    all_log_file = open(FILE_PATH_FOR_ALL_LOG, 'a')
    all_log_file.write(line_string)
    all_log_file.close()

    if elapsed_time > RESPONSE_TIME_WARINING_THRESHOLD:
        warning_log_file = open(FILE_PATH_FOR_WARNING_LOG, 'a')

        line_string = current_time + ' Warning!! Response time for hand recognition is slower than ' \
                      + str(RESPONSE_TIME_WARINING_THRESHOLD) + 's. Response Time = ' + str(elapsed_time) + '\n'

        print line_string
        warning_log_file.write(line_string)
        warning_log_file.close()

    # Check the validity of the response string
    match = re.search('"Status": "OK"', res)
    match_2 = re.search('"Gamjeong": 2001', res)
    if match and match_2:
        print "Valid response data"
    else:
        print "Invalid response data"
        warning_log_file = open(FILE_PATH_FOR_WARNING_LOG, 'a')

        line_string = current_time + ' Warning!! Invalid response detected from the hand response' + '\n'

        print line_string
        warning_log_file.write(line_string)
        warning_log_file.close()

    #print res

    time.sleep(REQUEST_SENDING_INTERVAL)

    ###########################################################################
    # Check response time for face recognition

    print '==================================================================='
    print 'face ip = ', face_rest_api_address[index]

    req = urllib2.Request(face_rest_api_address[index], data=face_image)
    req.add_header('Content-Length', '%d' % os.path.getsize(face_img_path))
    req.add_header('Content-Type', 'application/octet-stream')

    start_time = time.time()

    res = urllib2.urlopen(req, timeout=REQUEST_TIMEOUT_THRESHOLD_SEC).read().strip()

    elapsed_time = time.time() - start_time

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    line_string = current_time + ' Elapsed time(face) = ' + str(elapsed_time) + '\n'
    print line_string

    all_log_file = open(FILE_PATH_FOR_ALL_LOG, 'a')
    all_log_file.write(line_string)
    all_log_file.close()

    if elapsed_time > RESPONSE_TIME_WARINING_THRESHOLD:
        warning_log_file = open(FILE_PATH_FOR_WARNING_LOG, 'a')

        line_string = current_time + ' Warning!! Response time for face recognition is slower than ' \
                      + str(RESPONSE_TIME_WARINING_THRESHOLD) + 's. Response Time = ' + str(elapsed_time) + '\n'

        print line_string
        print '!!!!===================================================================!!!!'

        warning_log_file.write(line_string)
        warning_log_file.close()

    # Check the validity of the response string
    match = re.search('"Status": "OK"', res)
    match_2 = re.search('"eyebrows": 1010', res)
    if match and match_2:
        print "Valid response data"
    else:
        print "Invalid response data"
        warning_log_file = open(FILE_PATH_FOR_WARNING_LOG, 'a')

        line_string = current_time + ' Warning!! Invalid response detected from the face response' + '\n'
        print '!!!!===================================================================!!!!'
        print line_string
        warning_log_file.write(line_string)
        warning_log_file.close()

    #print res

    time.sleep(REQUEST_SENDING_INTERVAL)

    index = index + 1
    if index == 4:
        index = 0
