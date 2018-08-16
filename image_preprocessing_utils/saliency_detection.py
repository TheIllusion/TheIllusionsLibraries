#-*- coding: utf-8 -*-

import cv2
import numpy as np

#image = cv2.imread('/Users/Illusion/Data/deepfake_face_data/유재석_연예대상_face_cropped/6451.jpg', cv2.IMREAD_COLOR)
image = cv2.imread('/Users/Illusion/Data/stanford_dog_dataset/Images/n02086079-Pekinese/n02086079_207.jpg', cv2.IMREAD_COLOR)

options = ['spectral_residual', 'fine_grained']

selected_mode = options[1]

if selected_mode == options[0]:

    # initialize OpenCV's static saliency spectral residual detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imshow("Image", image)
    cv2.imshow("Output", saliencyMap)
    cv2.waitKey(0)

elif selected_mode == options[1]:

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = cv2.threshold(saliencyMap, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # show the images
    cv2.imshow("Image", image)
    #cv2.imshow("Output", saliencyMap)
    #cv2.imshow("Thresh", threshMap)

    concated_img = np.hstack((saliencyMap, threshMap))
    cv2.imshow("Result", concated_img)
    cv2.waitKey(0)