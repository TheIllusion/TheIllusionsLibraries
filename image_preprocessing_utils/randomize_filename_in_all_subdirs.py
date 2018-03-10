import os
import random

TARGET_DIR = '/Users/Illusion/Downloads/face_crop/_crawl'

for (path, dir, files) in os.walk(TARGET_DIR):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            rand_filename = "%d.jpg" % (random.randrange(0, 100000000))
            #print("rand_filename: %s" % rand_filename)
            print("path: %s" % path)
            #print("dir: %s" % dir)
            #print("filename: %s" % filename)

            os.rename(os.path.join(path,filename), os.path.join(path,rand_filename))