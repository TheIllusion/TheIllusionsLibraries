#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
	date=$1
	mkdir /data/users/rklee/server_hand_data/$date/
	#cp -ap /data_ssd/users/rklee/palmdemo_flask_8979/static/img/$date* /data/users/rklee/server_hand_data/$date/
	for file in /data_ssd/users/rklee/palmdemo_flask_8979/static/img/$date*; do cp "$file" /data/users/rklee/server_hand_data/$date/; done
	tar -cvf $1.tar $1/
fi