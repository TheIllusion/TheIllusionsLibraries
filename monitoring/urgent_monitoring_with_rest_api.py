import time
import os
import urllib2
import datetime
import re
import sys

REQUEST_SENDING_INTERVAL = 0.1
REQUEST_TIMEOUT_THRESHOLD_SEC = 30
RESPONSE_TIME_WARINING_THRESHOLD = 5

FILE_PATH_FOR_ALL_LOG = '/Users/Illusion/Temp/response_all_log.txt'
FILE_PATH_FOR_WARNING_LOG = '/Users/Illusion/Temp/response_warning.log'
FILE_PATH_FOR_ERROR_LOG = '/Users/Illusion/Temp/response_error.log'

#rest_api_address = 'http://vincent.nhnent.com:8979/hand'
#hand_rest_api_address = 'http://10.165.128.51:8979/hand'

hand_rest_api_address = []
face_rest_api_address = []

hand_rest_api_address.append('http://10.165.128.51:8979/hand-v2')
# hand_rest_api_address.append('http://10.161.31.22:8979/hand-v2')
# hand_rest_api_address.append('http://10.161.31.26:8979/hand-v2')
# hand_rest_api_address.append('http://10.161.31.27:8979/hand-v2')

face_rest_api_address.append('http://10.165.128.51:8989/face-v2')
# face_rest_api_address.append('http://10.161.31.22:8989/face-v2')
# face_rest_api_address.append('http://10.161.31.26:8989/face-v2')
# face_rest_api_address.append('http://10.161.31.27:8989/face-v2')

#hand_img_path = '/Users/Illusion/Documents/Palm_Data/random_hands/IMG_0703.jpg'

#12' Macbook
hand_img_path = '/Users/Illusion/Documents/rk_hand.jpg'
face_img_path = '/Users/Illusion/Documents/rk_face.jpg'

#Macbook Pro
#hand_img_path = '/Users/Illusion/Documents/Data/palm_data/test_set/crop_resize_512_512/rk.jpg'
#face_img_path = '/Users/Illusion/Documents/Data/toast_faces/rk.jpg'

f_hand = open(hand_img_path, 'rb')
hand_image = f_hand.read()
f_hand.close()

f_face = open(face_img_path, 'rb')
face_image = f_face.read()
f_face.close()

index = 0

hand_server_num = len(hand_rest_api_address)
face_server_num = len(face_rest_api_address)

print 'Total number of hand servers = ', str(hand_server_num)
print 'Total number of face servers = ', str(face_server_num)
while True:

    ###########################################################################
    # Check response time for hand recognition

    print '==================================================================='
    print 'hand ip = ', hand_rest_api_address[index]

    try:
        start_time = time.time()

        req = urllib2.Request(hand_rest_api_address[index], data=hand_image)
        req.add_header('Content-Length', '%d' % os.path.getsize(hand_img_path))
        req.add_header('Content-Type', 'application/octet-stream')

        res = urllib2.urlopen(req, timeout=REQUEST_TIMEOUT_THRESHOLD_SEC).read().strip()

        elapsed_time = time.time() - start_time

        current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

        line_string_all_log = current_time + ' Elapsed time(hand) = ' + str(elapsed_time) + '\n'
        print line_string_all_log

        all_log_file = open(FILE_PATH_FOR_ALL_LOG, 'a')
        all_log_file.write(line_string_all_log)
        all_log_file.close()

        if elapsed_time > RESPONSE_TIME_WARINING_THRESHOLD:
            warning_log_file = open(FILE_PATH_FOR_WARNING_LOG, 'a')

            line_string = current_time + ' Warning!!!!! Response time for hand recognition is slower than ' \
                          + str(RESPONSE_TIME_WARINING_THRESHOLD) + 's. Response Time = ' + str(elapsed_time) + hand_rest_api_address[index] + '\n'

            print line_string
            warning_log_file.write(line_string)
            warning_log_file.close()

        # Check the validity of the response string
        match = re.search('"Status": "OK"', res)
        match_2 = re.search('"Gamjeong": 2000', res)
        if match and match_2:
            print "Valid response data"
        else:
            print "Invalid response data"
            warning_log_file = open(FILE_PATH_FOR_WARNING_LOG, 'a')

            line_string = current_time + ' Warning!!!!! Invalid response detected from the hand response' + hand_rest_api_address[index] + '\n'

            print line_string
            warning_log_file.write(line_string)

            warning_log_file.close()

            # print res

    except:
        warning_log_file = open(FILE_PATH_FOR_ERROR_LOG, 'a')

        current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        line_string = current_time + ' Error!! Exception occurred in hand! ' + hand_rest_api_address[index] + '\n'

        print line_string
        warning_log_file.write(line_string)

        exception_string = "Unexpected error:" + str(sys.exc_info()[0]) + '\n'
        print exception_string
        warning_log_file.write(exception_string)

        warning_log_file.close()

    time.sleep(REQUEST_SENDING_INTERVAL)

    ###########################################################################
    # Check response time for face recognition

    print '==================================================================='
    print 'face ip = ', face_rest_api_address[index]

    try:
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

            line_string = current_time + ' Warning!!!!! Response time for face recognition is slower than ' \
                          + str(RESPONSE_TIME_WARINING_THRESHOLD) + 's. Response Time = ' + str(elapsed_time) + face_rest_api_address[index] + '\n'

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

            line_string = current_time + ' Warning!!!!! Invalid response detected from the face response' + face_rest_api_address[index] + '\n'
            print '!!!!===================================================================!!!!'
            print line_string
            warning_log_file.write(line_string)
            warning_log_file.close()

        #print res

    except:
        warning_log_file = open(FILE_PATH_FOR_ERROR_LOG, 'a')

        current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        line_string = current_time + ' Error!! Exception Occurred in face! ' + face_rest_api_address[index] + '\n'

        print line_string
        print '!!!!===================================================================!!!!'

        warning_log_file.write(line_string)

        exception_string = "Unexpected error:" + str(sys.exc_info()[0]) + '\n'
        print exception_string
        warning_log_file.write(exception_string)

        warning_log_file.close()

    time.sleep(REQUEST_SENDING_INTERVAL)

    index = index + 1
    if index == hand_server_num:
        index = 0
