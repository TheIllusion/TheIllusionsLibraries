import time
import os
import urllib2
import datetime

#rest_api_address = 'http://vincent.nhnent.com:8979/hand'
rest_api_address = 'http://10.165.128.51:8979/hand'

img_path = '/Users/Illusion/Documents/Data/palm_data/test_set/crop_resize_512_512'

os.chdir(img_path)

f = open(img_path + '/wr.jpg', 'rb')
fileContent = f.read()

#all_log_file = open('/Users/Illusion/Temp/response_all_log.txt', 'w')

while True:

    req = urllib2.Request(rest_api_address, data=fileContent)
    req.add_header('Content-Length', '%d' % os.path.getsize(img_path + '/wr.jpg'))
    req.add_header('Content-Type', 'application/octet-stream')

    start_time = time.time()

    res = urllib2.urlopen(req).read().strip()

    elapsed_time = time.time() - start_time

    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    line_string = current_time + ' Elapsed time = ' + str(elapsed_time) + '\n'
    print line_string
    #all_log_file.write(line_string)

    if elapsed_time > 0.3:
        warning_log_file = open('/Users/Illusion/Temp/response_warning_log.txt', 'w')
        line_string = current_time + ' Warning!! Response is slower than 0.4s. Response Time = ' + str(elapsed_time) + '\n'
        print line_string
        warning_log_file.write(line_string)
        warning_log_file.close()

    #print res

    time.sleep(10)

f.close()
#all_log_file.close()