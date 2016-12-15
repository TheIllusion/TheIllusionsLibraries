from subprocess import call
import datetime
import time

#Get the date of yesterday
timestamp = datetime.datetime.fromtimestamp(time.time() - 24*60*60).strftime('%Y_%m_%d')

print 'date = ', timestamp
call(["/home/nhnent/H1/users/rklee/Data/server_data/server_hand_data/get_data_from_svc.sh", timestamp])
