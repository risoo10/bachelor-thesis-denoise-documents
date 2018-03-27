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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import extract_join_patches


class Autoencoder:
    def __init__(self):
        self.model = 0
        self.train_noisy_patches = []
        self.train_residual_noise = []
        self.test_noisy_patches = []
        self.test_residual_noise = []
        self.batch_size = 128

    def load_data(self, filename):

        # Input data from file
        print("Loading data from file >", filename, "....")
        input = h5py.File(filename, "r")

        train_data = input["train_data"][:]
        validation_data = input["validation_data"][:]

        input.close()
        print("Loaded succesful ... |", train_data.shape, validation_data.shape)

        self.train_noisy_patches = train_data[1]
        self.train_residual_noise = train_data[0]
        self.test_noisy_patches = validation_data[1]
        self.test_residual_noise = validation_data[0]

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
        self.model.fit(self.train_noisy_patches, self.train_residual_noise,
                       epochs=epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=(self.test_noisy_patches, self.test_residual_noise),
                       callbacks=[TensorBoard(log_dir="logs/"+filename, histogram_freq=0, write_graph=True)])

        self.model.save("models/" + filename + ".h5")

    def predict_from_image(self, image_name):
        width, height, channels = 16, 12, 3
        patch_size = 64
        step = 32

        # Load Image
        test_image = cv2.imread(image_name)
        height, width = test_image.shape[:2]
        print("Loaded image | Shape:", test_image.shape)

        # Resize and normalize image
        width = int(width / (2 * step)) * step
        height = int(height / (2 * step)) * step
        test_image = cv2.resize(test_image, (width, height))
        height, width = test_image.shape[:2]
        test_image = test_image.astype("float32") / 255
        print("Resized image | Shape:", test_image.shape)

        # Extract patches
        patches = extract_join_patches.split_to_patches(test_image, shape=(patch_size, patch_size, channels), step=step)
        patch_count_ver, patch_count_hor = patches.shape[:2]

        # Flatten to 1D array of patches
        patches = patches.reshape((patch_count_ver * patch_count_hor, patches.shape[3], patches.shape[4], patches.shape[5]))

        # Predict results
        denoised_patches = self.model.predict(patches)

        # Reshape back to 2D array of patches
        denoised_patches = denoised_patches.reshape((patch_count_ver, patch_count_hor, patches.shape[1], patches.shape[2], patches.shape[3]))

        # Reconstruct final image from patches
        reconstructed = extract_join_patches.patch_together(denoised_patches, image_size=(width, height))

        # Count Score
        # score = ((test_image - reconstructed) ** 2).mean(axis=None)

        # Plot noisy and denoised images
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
        # fig.suptitle("MSE: " + str(score))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(reconstructed.astype("float32"), cv2.COLOR_BGR2RGB))
        plt.show()

    def predict_test_patches(self, filename):

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
autoencoder.predict_from_image("test_images/001.bmp")

# Check available GPU
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Preprocess data and train
# autoencoder.load_and_preprocess_data(DATA_FILE)
# autoencoder.compile_model()
# autoencoder.train_model('conv-autoencoder-renoir-64x64-MEDIUM-DP-MSE-SMLR', epochs=150)
# autoencoder.predict_test_patches()
