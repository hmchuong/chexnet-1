from utils import extract_data, check_and_create_dir, resize, crop, preprocess
import os
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave, imresize
import imreg_dft as ird
import numpy as np

class ImageProcessing:
    """ImageProcessing for registration, augmentation images"""
    def __init__(self, image_size):
        self.image_size = image_size

    def augmentation(self, input_dir, output_dir, seed):
        """
        Augmentation data
        """
        # Input
        x_images = extract_data([input_dir])
        print(x_images)
        # Output
        x_path = output_dir

        train = crop(x_images, upsampling='True')
        train = resize(train, self.image_size)
        train = preprocess(train)
        trainX = np.reshape(train, (len(train), self.image_size, self.image_size, 1))
        data_gen_args = dict(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=5.,
                    width_shift_range=0.08,
                    height_shift_range=0.08,
                    shear_range=0.06,
                    zoom_range=0.08,
                    channel_shift_range=0.2,
                    fill_mode='constant',
                    cval=0.,
                    horizontal_flip=True,
                    vertical_flip=False,
                    rescale=None)
        image_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(trainX, augment=True, seed=1)
        batch_size = len(trainX)
        print("Batch_size",batch_size)
        for i in range(seed):
            print("Generate seed: {}/{}".format(i+1,seed))
            x = image_datagen.flow(trainX, shuffle=True, seed=i, save_format='png', save_to_dir=x_path, batch_size=batch_size)
            _ = x.next()
