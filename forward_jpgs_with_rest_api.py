from subprocess import call
import os
import glob

# (USE EXAMPLE) curl -H "Content-type: application/octet-stream" http://vincent.nhnent.com:8979/hand --data-binary @jh.jpg

rest_api_address = 'http://vincent.nhnent.com:8979/hand'
content_type = 'Content-type: application/octet-stream'
img_path = '/Users/Illusion/Documents/Data/palm_data/test_set/crop_resize_512_512'

os.chdir(img_path)

jpg_files = glob.glob( '*.jpg' )

for jpg_file in jpg_files:
    file_name = jpg_file[0:]
    file_string = '@' + file_name
    print '#######################################################################'
    call( ["curl", '-H', content_type, rest_api_address, '--data-binary', file_string] )
    print 'input filename: ', file_string

JPG_files = glob.glob( '*.JPG' )

for JPG_file in JPG_files:
    file_name = JPG_file[0:]
    file_string = '@' + file_name
    print '#######################################################################'
    call(["curl", '-H', content_type, rest_api_address, '--data-binary', file_string])
    print 'input filename: ', file_string

print 'End'