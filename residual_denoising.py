import h5py
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Activation, BatchNormalization
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D
from keras import optimizers
from keras.callbacks import TensorBoard


class ResidualLearning:
    def __init__(self):
        self.model = 0
        self.train_noisy_patches = []
        self.train_patches = []
        self.test_noisy_patches = []
        self.test_patches = []
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
        self.train_patches = train_data[0]
        self.test_noisy_patches = validation_data[1]
        self.test_patches = validation_data[0]

    def compile_model(self):
        # residual convolutional net archtecture
        input_img = Input(shape=(64, 64, 3))

        convnet = Conv2D(64, (3, 3), padding='same')(input_img)
        convnet = Activation('relu')(convnet)

        # Set the depth of the residual hidden layers
        depth_hidden_layers = 12

        for i in range(depth_hidden_layers):
            convnet = Conv2D(64, (3, 3), padding='same')(convnet)
            convnet = BatchNormalization()(convnet)
            convnet = Activation('relu')(convnet)

        resConvnet = Conv2D(3, (3, 3), padding="same")(convnet)

        self.model = Model(input_img, resConvnet)
        self.model.compile(optimizer="adam", loss='mse', metrics=["acc"])


    def load_model_from_file(self, filename):
        print('Loading model from file ... ... ...')
        self.model = load_model("models/" + filename + ".h5")
        print('Model successfully loaded from file.')

    def train_model(self):
        pass

    def predict_test_images(self, filename):
        pass