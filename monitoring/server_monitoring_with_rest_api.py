import time
import os, subprocess

rest_api_address = 'http://vincent.nhnent.com:8979/hand'
content_type = 'Content-type: application/octet-stream'
img_path = '/Users/Illusion/Documents/Data/palm_data/test_set/crop_resize_512_512'

os.chdir(img_path)

file_string = '@' + "wr.jpg"
subprocess.call( ["curl", '-H', content_type, rest_api_address, '--data-binary', file_string] )