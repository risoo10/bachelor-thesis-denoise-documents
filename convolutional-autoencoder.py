import keras
import cv2
import h5py
import sklearn
from time import time
import random

import numpy as np
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Activation, BatchNormalization
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D
from keras import optimizers
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
        self.batch_size = 128

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

        encode = Conv2D(64, (3, 3), padding='same')(input_img)
        # encode = BatchNormalization()(encode)
        encode = Activation('relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        encode = Conv2D(32, (3, 3), padding='same')(encode)
        # encode = BatchNormalization()(encode)
        encode = Activation('relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        encode = Conv2D(32, (3, 3), padding='same')(encode)
        encode = Activation('relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)
        encode = Dropout(0.2)(encode)

        decode = Conv2D(32, (3, 3), padding='same')(encode)
        # decode = BatchNormalization()(decode)
        decode = Activation('relu')(decode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(32, (3, 3), padding='same')(decode)
        # decode = BatchNormalization()(decode)
        decode = Activation('relu')(decode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(64, (3, 3), padding='same')(decode)
        # decode = BatchNormalization()(decode)
        decode = Activation('relu')(decode)
        decode = UpSampling2D((2, 2))(decode)
        decode = Dropout(0.2)(decode)

        decode = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(decode)

        self.model = Model(input_img, decode)
        self.compile = self.model.compile(optimizer=optimizers.adam(lr=0.0001), loss='mse', metrics=["acc"])

    def load_model_from_file(self, filename='autoencoder.h5'):
        print('Loading model from file ... ... ...')
        self.model = load_model("models/" + filename + ".h5")
        print('Model successfully loaded from file.')

    def train_model(self, filename='autoencoder', epochs=100):
        self.model.fit(self.train_noisy_patches, self.train_patches,
                       epochs=epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=(self.test_noisy_patches, self.test_patches),
                       callbacks=[TensorBoard(log_dir="logs/"+filename, histogram_freq=0, write_graph=True)])

        self.model.save("models/" + filename + ".h5")

    def predict_full_size_images(self, filename):

        print("Loading data from file >", filename, "....")
        input = h5py.File(filename, "r")

        test_data = input["test_data"][:]

        input.close()

        random_sed = random.randrange(0, 100)
        test_data[0], test_data[1] = sklearn.utils.shuffle(test_data[0], test_data[1], random_state=random_sed)

        clean_images = test_data[0][:100]
        noisy_images = test_data[1][:100]



        score = self.model.evaluate(noisy_images, clean_images, batch_size=10)
        print(score)

        denoised_images = self.model.predict(noisy_images)

        n = 5
        plt.figure(figsize=(18, 15))
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


    def add_gaussian_noise(img, mean=0, var=0.01):
        # Add noise
        sigma = var ** 0.5
        result = img + np.random.normal(mean, sigma, img.shape)
        return np.clip(result, 0., 1.)


DATA_FILE = "renoir-Mi3-aligned-64x64-CLEAN-NOISY.h5"
autoencoder = Autoencoder()


# Load the saved model and predict images
autoencoder.load_model_from_file(filename="conv-autoencoder-renoir-64x64-MEDIUM-DP-MSE-SMLR")
autoencoder.predict_full_size_images(DATA_FILE)

# Check available GPU
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Preprocess data and train
# autoencoder.load_and_preprocess_data(DATA_FILE)
# autoencoder.compile_model()
# autoencoder.train_model('conv-autoencoder-renoir-64x64-MEDIUM-DP-MSE-SMLR', epochs=150)
# autoencoder.predict_full_size_images()
