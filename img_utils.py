import numpy as np
from scipy.misc import imsave, imread, imresize
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d

def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    '''x shape: (num_channels, rows, cols)'''
    height, width = x.shape[:2]
    if upscale: x = imresize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def combine_patches(in_patches, out_shape, scale):
    '''Reconstruct an image from these `patches`'''
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon

def subimage_generator(img, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, IMG_SHAPE - patch_size, stride):
            for y in range(0, IMG_SHAPE - patch_size, stride):
                subimage = img[x : x + patch_size, y : y + patch_size, :]

                yield subimage
