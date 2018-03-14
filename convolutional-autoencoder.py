import keras
import cv2
import h5py
from time import time

import numpy as np
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Activation
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D
from keras.callbacks import TensorBoard
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(self):
        self.model = 0
        self.train_noisy_patches = []
        self.train_patches = []
        self.test_noisy_patches = []
        self.test_patches = []

    def load_and_preprocess_data(self, filename):


        # Input data from file
        print("Loading data from file >", filename, "....")
        input = h5py.File(filename, "r")

        train_data = input["train_data"][:]
        validation_data = input["validation_data"][:]

        input.close()
        print("Loaded succesful ... |", train_data.shape, validation_data.shape)

        self.train_noisy_patches = train_data[1]
        self.train_patches = train_data[0]
        self.test_noisy_patches = validation_data[1]
        self.test_patches = validation_data[0]

    def compile_model(self):
        # Convolutional autoencoder architecture
        input_img = Input(shape=(64, 64, 3))

        encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        encode = Conv2D(32, (3, 3), padding='same', activation='relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        encode = Conv2D(16, (3, 3), padding='same', activation='relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        decode = Conv2D(16, (3, 3), padding='same', activation='relu')(encode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(32, (3, 3), padding='same', activation='relu')(decode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(64, (3, 3), padding='same', activation='relu')(decode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(decode)

        self.model = Model(input_img, decode)

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def load_model_from_file(self, filename='autoencoder.h5'):
        print('Loading model from file ... ... ...')
        self.model = load_model(filename)
        print('Model successfully loaded from file.')

    def train_model(self, filename='autoencoder.h5', epochs=100):
        self.model.fit(self.train_noisy_patches, self.train_patches,
                       epochs=epochs,
                       batch_size=128,
                       shuffle=True,
                       validation_data=(self.test_noisy_patches, self.test_patches),
                       callbacks=[TensorBoard(log_dir="logs/final/{}".format(time()), histogram_freq=1, write_graph=True, write_images=True)])

        self.model.save(filename)

    def predict_full_size_images(self):

        n = 10
        clean_images = self.test_patches[:-n]
        noisy_images = self.test_noisy_patches[:-n]
        denoised_images = self.model.predict(noisy_images)

        plt.figure(figsize=(20, 6))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i+1)
            plt.imshow(cv2.cvtColor(clean_images[i], cv2.COLOR_BGR2RGB))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noise
            ax = plt.subplot(3, n, i + n + 1)
            plt.imshow(cv2.cvtColor(noisy_images[i], cv2.COLOR_BGR2RGB))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display result
            ax = plt.subplot(3, n, i + 2 * n + 1)
            plt.imshow(cv2.cvtColor(denoised_images[i], cv2.COLOR_BGR2RGB))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()



        self.model.evaluate()

    def add_gaussian_noise( img, mean=0, var=0.01):
        # Add noise
        sigma = var ** 0.5
        result = img + np.random.normal(mean, sigma, img.shape)
        return np.clip(result, 0., 1.)


DATA_FILE = "renoir-Mi3-aligned-64x64-CLEAN-NOISY.h5"
autoencoder = Autoencoder()


# Load the saved model and predict images
# autoencoder.load_model_from_file(filename="autoencoder-mnist-16x16.h5")
# autoencoder.predict_full_size_images()

# Check available GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Preprocess data and train
autoencoder.load_and_preprocess_data(DATA_FILE)
autoencoder.compile_model()
autoencoder.train_model('conv-autoencoder-renoir-64x64.h5', epochs=5)
autoencoder.predict_full_size_images()
