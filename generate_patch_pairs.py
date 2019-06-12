#!/usr/bin/python

import os
import cv2
import numpy as np
from sklearn.utils import shuffle

PATCH_SIZE = 9 #8 #9 #128
STRIDE = 9 #8 #9 #128
SAMPLING = 4

DATASET_ROOT = '/home/go/work/research/colorization/data/Train_gray'
OUT_PATH = '{0}/../train_{1}_{2}_{3}.npz'.format(DATASET_ROOT,PATCH_SIZE,STRIDE,SAMPLING)

x_train_gray = []
x_train_color = []

count_img = 0
count_total_patch = 0
for root,dirs,files in os.walk(os.path.join(DATASET_ROOT)):
    for f in files:
        # ignore files expect png files
        if not f.endswith('png'):
            continue
        count_img += 1
        gray_image = cv2.imread(os.path.join(root,f))
        color_image = cv2.imread(os.path.join(root,f).replace('Train_gray','DIV2K_train_HR'))
        # block
        #gray_image = gray_image.astype('float32') / 255.
        #color_image = color_image.astype('float32') / 255.
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        #gray_image = np.zeros(gray_image.shape,dtype='uint8')
        #for c in range(color_image.shape[2]):
        #    color_image[:,:,c] = color_image[:,:,2]
        #color_image /= 128.
        # block
        count_patch = 0
        for y in xrange(0, gray_image.shape[0], STRIDE):
            for x in xrange(0, gray_image.shape[1], STRIDE):
                gray_patch = gray_image[y:y+PATCH_SIZE,x:x+PATCH_SIZE,:]
                color_patch = color_image[y:y+PATCH_SIZE,x:x+PATCH_SIZE,:]
                if gray_patch.shape[0:2] != (PATCH_SIZE,PATCH_SIZE) or \
		   color_patch.shape[0:2] != (PATCH_SIZE,PATCH_SIZE):
                    continue
                count_patch += 1
                count_total_patch += 1
                if count_total_patch % SAMPLING != 0:
                    continue
                x_train_gray.append(gray_patch)
                x_train_color.append(color_patch)
                print 'Processed ' + str(count_patch) + ' / ' + str(count_img)

print 'Total number of patch:',str(count_total_patch)

x_train_gray = np.array(x_train_gray)
x_train_color = np.array(x_train_color)

# shuffling
x_train_gray,x_train_color = shuffle(x_train_gray,x_train_color)

print('Saving..')
np.savez(OUT_PATH, x_train_gray, x_train_color)
print('Saved..')
