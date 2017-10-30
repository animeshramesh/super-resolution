'''
Dump all your images in dataset/output.
This script will generate the corresponding low-res images in dataset/input.
1. Apply Gaussian blur
2. Downsample images
'''

import os
import numpy
import scipy
import scipy.misc, scipy.ndimage


INPUT_DIR = 'dataset/output'
OUTPUT_DIR = 'dataset/input'
DOWNSAMPLE_RATIO = 0.25

for img_file in os.listdir(INPUT_DIR):
    img = scipy.misc.imread(os.path.join(INPUT_DIR, img_file))
    blurred = scipy.ndimage.filters.gaussian_filter(img, sigma=(3, 3, 0))
    downsampled_img = scipy.misc.imresize(blurred, DOWNSAMPLE_RATIO)
    scipy.misc.imsave(os.path.join(OUTPUT_DIR, img_file), downsampled_img)
