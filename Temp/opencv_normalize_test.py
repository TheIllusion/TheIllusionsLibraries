# normalize pixel values to pytorch style

'''
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
'''

import cv2

# Input Image
filename = "/Users/Illusion/Documents/rk_face.jpg"

img = cv2.imread(filename, cv2.IMREAD_COLOR)

img = img.astype(float)

img = img / 255.0

img[..., 0] = (img[..., 0] - 0.406) / 0.225
img[..., 1] = (img[..., 1] - 0.456) / 0.224
img[..., 2] = (img[..., 2]  - 0.485) / 0.229

#print img.shape

print img[0][0][0]
print img[1][1][0]
print img[2][2][0]
print img[3][3][0]