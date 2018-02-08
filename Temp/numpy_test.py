import numpy as np

BATCH_SIZE = 3

hair_color_list = ['blue', 'brown', 'red', 'blonde', 'wine']

condition_vectors = np.zeros(shape=(BATCH_SIZE, len(hair_color_list), 50, 50))
#condition_vectors = np.empty(shape=(BATCH_SIZE, len(hair_color_list), 50, 50))

condition_vectors[:,2,:,:] = 100

condition_vectors[:,:,:,:] = 0

print condition_vectors.shape