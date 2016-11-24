import time
import os
import urllib2
import datetime

#rest_api_address = 'http://vincent.nhnent.com:8979/hand'
hand_rest_api_address = 'http://10.165.128.51:8979/hand'
face_rest_api_address = 'http://10.165.128.51:8989/face'

hand_img_path = '/Users/Illusion/Documents/Data/palm_data/test_set/crop_resize_512_512'
face_img_path = '/Users/Illusion/Documents/Data/toast_faces'

f_hand = open(hand_img_path + '/rk.jpg', 'rb')
hand_image = f_hand.read()
f_hand.close()

f_face = open(face_img_path + '/rk.jpg', 'rb')
face_image = f_face.read()
f_face.close()

#all_log_file = open('/Users/Illusion/Temp/response_all_log.txt', 'w')

while True:

    # Check response time for hand recognition
    req = urllib2.Request(hand_rest_api_address, data = hand_image)
    req.add_header('Content-Length', '%d' % os.path.getsize(hand_img_path + '/rk.jpg'))
    req.add_header('Content-Type', 'application/octet-stream')

    start_time = time.time()

    res = urllib2.urlopen(req).read().strip()

    elapsed_time = time.time() - start_time

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    line_string = current_time + ' Elapsed time(hand) = ' + str(elapsed_time) + '\n'
    print line_string
    #all_log_file.write(line_string)

    if elapsed_time > 0.3:
        warning_log_file = open('/Users/Illusion/Temp/response_warning.log', 'a')
        line_string = current_time + ' Warning!! Response time for hand recognition is slower than 0.4s. Response Time = ' + str(elapsed_time) + '\n'
        print line_string
        warning_log_file.write(line_string)
        warning_log_file.close()

    #print res

    time.sleep(10)

    # Check response time for face recognition
    req = urllib2.Request(face_rest_api_address, data=face_image)
    req.add_header('Content-Length', '%d' % os.path.getsize(face_img_path + '/rk.jpg'))
    req.add_header('Content-Type', 'application/octet-stream')

    start_time = time.time()

    res = urllib2.urlopen(req).read().strip()

    elapsed_time = time.time() - start_time

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    line_string = current_time + ' Elapsed time(face) = ' + str(elapsed_time) + '\n'
    print line_string
    # all_log_file.write(line_string)

    if elapsed_time > 0.3:
        warning_log_file = open('/Users/Illusion/Temp/response_warning.log', 'a')
        line_string = current_time + ' Warning!! Response time for face recognition is slower than 0.4s. Response Time = ' + str(
            elapsed_time) + '\n'
        print line_string
        warning_log_file.write(line_string)
        warning_log_file.close()

    #print res

    time.sleep(10)


#all_log_file.close()