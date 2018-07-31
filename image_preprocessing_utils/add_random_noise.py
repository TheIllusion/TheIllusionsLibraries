# source copied from 'https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv?rq=1'

'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
'''

import numpy as np
import os
import cv2

def noisy(noise_typ, image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.4
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

if __name__ == '__main__':
    #img = cv2.imread('./mask_rect.png', cv2.IMREAD_UNCHANGED)
    img = cv2.imread('/Users/Illusion/Documents/Data/test_faces/Test_20/1346833132858.jpg', cv2.IMREAD_COLOR)

    if type(img) is np.ndarray:
        img = cv2.resize(img, (300, 300), cv2.INTER_LINEAR)

        result_gaussian = noisy('gauss', img.copy())
        result_salt_pepper = noisy('s&p', img.copy())
        result_poisson = noisy('poisson', img.copy())
        result_speckle = noisy('speckle', img.copy())

        # stack images vertically(axis 0) or horizontally(axis 1)
        #vis = np.concatenate((img.copy(), result_gaussian, result_salt_pepper, result_poisson, result_speckle), axis=0)
        #cv2.imshow('results', vis)

        print 'img.shape=', img.shape
        print 'result_gaussian.shape=', result_gaussian.shape
        print 'result_salt_pepper.shape=', result_salt_pepper.shape
        print 'result_poisson.shape=', result_poisson.shape
        print 'result_speckle.shape=', result_speckle.shape

        #cv2.imshow('original', img.copy())
        #cv2.imshow('result_gaussian', result_gaussian)
        cv2.imshow('result_salt_pepper', result_salt_pepper)
        cv2.imshow('result_salt_pepper2', noisy('s&p', img.copy()))
        cv2.imshow('result_salt_pepper3', noisy('s&p', img.copy()))
        cv2.imshow('result_salt_pepper4', noisy('s&p', img.copy()))
        cv2.imshow('result_salt_pepper5', noisy('s&p', img.copy()))

        #cv2.imshow('result_poisson', result_poisson)
        #cv2.imshow('result_speckle', result_speckle)

        cv2.waitKey()

    print 'process end'