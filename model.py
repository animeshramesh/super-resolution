import os
from scipy.misc import imsave, imread, imresize
import pdb
import numpy as np
import img_utils

import keras
from keras.models import Model, load_model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def predict(model, img, scale_factor=2, patch_size=64):
    init_width, init_height = img.shape[0], img.shape[1]

    # Naive upsampling
    baseline = imresize(img, (init_width * scale_factor, init_height * scale_factor))

    # Split into patches
    images = img_utils.make_patches(img, scale_factor, patch_size, True, 1)

    # Predict HR image using model
    img_conv = images.astype(np.float32) / 255.
    result = model.predict(img_conv, batch_size=128, verbose=True)
    result = result.astype(np.float32) * 255.
    out_shape = (init_width * scale_factor, init_height * scale_factor, 3)
    result = img_utils.combine_patches(result, out_shape, scale_factor)

    return result, baseline

# Constants
SCALE_FACTOR = 2
IMG_SHAPE = 256          # must be divisible by stride
STRIDE = 16
LR_PATCH_SIZE = 32
HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE_FACTOR

# Get training data
X_DIR = 'dataset/X'
Y_DIR = 'dataset/Y'

X_train = []
Y_train = []
for img_file in os.listdir(X_DIR):
    if img_file.endswith('.png'):
        x = imread(os.path.join(X_DIR, img_file))
        y = imread(os.path.join(Y_DIR, img_file))
        X_train.append(x)
        Y_train.append(y)

# Define the model and train
inp = Input(shape=(HR_PATCH_SIZE, HR_PATCH_SIZE, 3))
x = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same')(inp)
x = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same')(inp)
x = Convolution2D(filters=3, kernel_size=3, padding='same')(inp)

model = Model(inputs=inp, outputs=x)
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(np.array(X_train),
          np.array(Y_train),
          epochs=10,
          batch_size=32,
          shuffle=True,
          verbose=1,
          validation_split=0.05)
model.save('srcnn.h5')

model = load_model('srcnn.h5')

# Predict on sample img
img = imread('/Users/admin/Downloads/images_all/Philadelphia_Vireo_0027_2889358328.jpg')
img = imresize(img, (IMG_SHAPE/SCALE_FACTOR, IMG_SHAPE/SCALE_FACTOR))
prediction, baseline = predict(model, img)
imsave('/Users/admin/Downloads/prediction.png', prediction)
imsave('/Users/admin/Downloads/baseline.png', baseline)
imsave('/Users/admin/Downloads/input.png', img)
