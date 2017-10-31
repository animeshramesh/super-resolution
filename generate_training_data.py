'''
Dump all your images in dataset/output.
This script will generate the corresponding low-res images in dataset/input.
1. Apply Gaussian blur
2. Downsample images
'''

import os
import numpy as np
import scipy
import scipy.misc, scipy.ndimage
from scipy.misc import imsave, imread, imresize
from scipy.ndimage.filters import gaussian_filter
import tqdm
import time

# Paths
DATASET_DIR = '/Users/admin/Downloads/images_all/'
TARGET_DIR = '/Users/admin/Dev/super-resolution/dataset/'

# Constants
SCALE_FACTOR = 2
IMG_SHAPE = 256          # must be divisible by stride
STRIDE = 16
LR_PATCH_SIZE = 32
HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE_FACTOR

# Create target directory if not present
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
if not os.path.exists(os.path.join(TARGET_DIR, 'Y')):
    os.makedirs(os.path.join(TARGET_DIR, 'Y'))
if not os.path.exists(os.path.join(TARGET_DIR, 'X')):
    os.makedirs(os.path.join(TARGET_DIR, 'X'))

def subimage_generator(img, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, IMG_SHAPE - patch_size, stride):
            for y in range(0, IMG_SHAPE - patch_size, stride):
                subimage = img[x : x + patch_size, y : y + patch_size, :]

                yield subimage

for index, img_file in enumerate(os.listdir(DATASET_DIR)):
    img = imread(os.path.join(DATASET_DIR, img_file), mode='RGB')

    # Resize to 256 x 256
    img = imresize(img, (IMG_SHAPE, IMG_SHAPE))

     # Create patches
    nb_hr_images = (IMG_SHAPE ** 2) // (STRIDE ** 2)    # Flooring division
    hr_samples = np.empty((nb_hr_images, HR_PATCH_SIZE, HR_PATCH_SIZE, 3))
    image_subsample_iterator = subimage_generator(img, STRIDE, HR_PATCH_SIZE, nb_hr_images)

    stride_range = np.sqrt(nb_hr_images).astype(int)

    i = 0
    for j in range(stride_range):
        for k in range(stride_range):
            hr_samples[i, :, :, :] = next(image_subsample_iterator)
            i += 1

    t1 = time.time()
    # Create nb_hr_images 'X' and 'Y' sub-images of size hr_patch_size for each patch
    for i in range(nb_hr_images):
        ip = hr_samples[i]

        # Save ground truth image Y
        imsave(TARGET_DIR + "/Y/" + "%d_%d.png" % (index + 1, i + 1), ip)

        # Apply Gaussian Blur to Y
        op = gaussian_filter(ip, sigma=0.5)

        # Subsample by scaling factor to create X
        op = imresize(op, (LR_PATCH_SIZE, LR_PATCH_SIZE), interp='bicubic')

        # Save X
        imsave(TARGET_DIR + "/X/" + "%d_%d.png" % (index + 1, i+1), op)

    print("Finished image %d in time %0.2f seconds.. " % (index + 1, time.time() - t1))
