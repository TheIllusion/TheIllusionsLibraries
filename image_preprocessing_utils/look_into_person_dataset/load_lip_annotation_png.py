import cv2, glob

lip_annotation_pngs = '/Users/Illusion/Temp/*.png'

lib_annotation_filelist = glob.glob(lip_annotation_pngs)

for png in lib_annotation_filelist:
    img = cv2.imread(png, cv2.IMREAD_UNCHANGED)

    idx = (img[...] > 17)

    print 'filename =', png
    print img[idx]

