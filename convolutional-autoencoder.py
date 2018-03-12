import keras
import cv2

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

    def load_and_preprocess_data(self):
        # Input data
        (x_train, _), (x_test, _) = cifar10.load_data()

        # Normalize data
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.


        # Change to greyscale
        w = np.array([0.21, 0.72, 0.07])

        x_train = x_train * w
        x_test = x_test * w

        x_train = np.sum(x_train, axis=3)
        x_test = np.sum(x_test, axis=3)


        # One channel conversion channel first
        x_train = np.reshape(x_train, (len(x_train), 32, 32, 1))
        x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))

        # Plot sample images
        # fig = plt.figure(figsize=(6, 2))
        # for i in range(4):
        #     # display original
        #     ax = plt.subplot(1, 4, i + 1)
        #     index = randrange(0, len(x_train))
        #     plt.imshow(x_train[index].reshape(28, 28))
        #     plt.gray()
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        # plt.show()

        # Generate small patches
        # print('Generating patches from images ... ... ...')
        #
        # x_train_patches = image.PatchExtractor((16, 16), 0.01).transform(x_train)
        # x_train_patches = np.reshape(x_train_patches, (len(x_train_patches), 16, 16, 1))
        #
        # x_test_patches = image.PatchExtractor((16, 16), 0.01).transform(x_test)
        # x_test_patches = np.reshape(x_test_patches, (len(x_test_patches), 16, 16, 1))
        #
        # print('Finished generating patches !')

        # Plot sample patches
        # fig = plt.figure(figsize=(9, 2))
        # for i in range(10):
        #     # display original
        #     ax = plt.subplot(1, 10, i + 1)
        #     plt.imshow(x_train_patches[i].reshape(16, 16))
        #     plt.gray()
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        # plt.show()
        # # fig.savefig("sample_patches.png")

        # Use full size images

        x_train_patches = x_train
        x_test_patches = x_test

        x_train_patches_noise = Autoencoder.add_gaussian_noise(x_train_patches, mean=0, var=0.01)
        x_test_patches_noise = Autoencoder.add_gaussian_noise(x_test_patches, mean=0, var=0.01)

        # Plot damaged patches with noise
        # fig = plt.figure(figsize=(9, 2))
        # for i in range(10):
        #     # display original
        #     ax = plt.subplot(1, 10, i + 1)
        #     plt.imshow(x_train_patches_noise[i].reshape(16, 16))
        #     plt.gray()
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        # plt.show()
        # # fig.savefig("sample_patches.png")

        self.train_noisy_patches = x_train_patches_noise
        self.train_patches = x_train_patches
        self.test_noisy_patches = x_test_patches_noise
        self.test_patches = x_test_patches

    def compile_model(self):
        # Convolutional autoencoder architecture

        input_img = Input(shape=(None, None, 1))

        encode = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        encode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
        encode = MaxPooling2D((2, 2), padding='same')(encode)

        decode = Conv2D(64, (3, 3), padding='same', activation='relu')(encode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(64, (3, 3), padding='same', activation='relu')(decode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(64, (3, 3), padding='same', activation='relu')(decode)
        decode = UpSampling2D((2, 2))(decode)

        decode = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(decode)

        self.model = Model(input_img, decode)

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def load_model_from_file(self, filename='autoencoder.h5'):
        print('Loading model from file ... ... ...')
        self.model = load_model(filename)
        print('Model successfully loaded from file.')

    def train_model(self, filename='autoencoder.h5'):
        self.model.fit(self.train_noisy_patches, self.train_patches,
                       epochs=50,
                       batch_size=128,
                       shuffle=True,
                       validation_data=(self.test_noisy_patches, self.test_patches),
                       callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

        self.model.save(filename)

    def predict_full_size_images(self):


        n = 10
        size = 32
        images = self.test_patches[:-n]

        noisy_images = Autoencoder.add_gaussian_noise(images, mean=0, var=0.01)

        denoised_images = self.model.predict(noisy_images)


        plt.figure(figsize=(20, 6))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i+1)
            plt.imshow(images[i].reshape(size, size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display noise
            ax = plt.subplot(3, n, i + n + 1)
            plt.imshow(noisy_images[i].reshape(size, size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display result
            ax = plt.subplot(3, n, i + 2 * n + 1)
            plt.imshow(denoised_images[i].reshape(size, size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

        self.model.evaluate()


    @staticmethod
    def add_gaussian_noise( img, mean=0, var=0.01):
        # Add noise
        sigma = var ** 0.5
        result = img + np.random.normal(mean, sigma, img.shape)
        return np.clip(result, 0., 1.)


autoencoder = Autoencoder()


# Load the saved model and predict images
autoencoder.load_model_from_file(filename="autoencoder-mnist-16x16.h5")
autoencoder.predict_full_size_images()

# Preprocess data and train
# autoencoder.load_and_preprocess_data()
# autoencoder.compile_model()
# autoencoder.train_model('autoencoder-cifar10-32x32')
# autoencoder.predict_full_size_images()
