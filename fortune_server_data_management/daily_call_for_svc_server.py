from subprocess import call
import datetime
import time

#Get the date of yesterday
timestamp = datetime.datetime.fromtimestamp(time.time() - 24*60*60).strftime('%Y_%m_%d')

print 'date = ', timestamp
call(["/data/users/rklee/server_hand_data/copy_files_from_ssd.sh", timestamp])
