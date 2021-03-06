#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
  date=$1

  wget http://10.161.31.22:8985/server_face_data/$date.tar
  tar -xvf $date.tar
  mv $date.tar $date.svc001.tar

  wget http://10.161.31.24:8985/server_face_data/$date.tar
  tar -xvf $date.tar
  mv $date.tar $date.svc003.tar

  wget http://10.161.31.25:8985/server_face_data/$date.tar
  tar -xvf $date.tar
  mv $date.tar $date.trn001.tar

  wget http://10.161.31.23:8985/server_face_data/$date.tar
  tar -xvf $date.tar
  mv $date.tar $date.svc002.tar

  wget http://10.161.31.26:8985/server_face_data/$date.tar
  tar -xvf $date.tar
  mv $date.tar $date.svc004.tar

  wget http://10.161.31.27:8985/server_face_data/$date.tar
  tar -xvf $date.tar
  mv $date.tar $date.svc005.tar

  cd $date
  #erase rk.jpg
  #find . -size 182936c -exec rm -rf {} \;
  #erase output_rk.jpg
  #find . -size 83815c -exec rm -rf {} \;

  for file in $date*; do cp "$file" /home/nhnent/H2/users/mskang/web_result/server_data/original_resized_face/; done
  for file in output_$date*; do cp "$file" /home/nhnent/H2/users/mskang/web_result/server_data/result_face/; done

  cd /home/nhnent/H2/users/mskang/web_result/server_data/

  python delete_duplicate_file.py $date
  python update_file_list_face.py
  python update_guest_file_list.py

fi
