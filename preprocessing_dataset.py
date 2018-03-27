import os
import numpy as np
import sys
import h5py
import cv2
import sklearn.utils
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt


filepath = "D:\RISKO\SKOLA\V-semester\Bakalarka\Datasets\Renoir\Mi3_Aligned\Mi3_Aligned"

RANDOM_SEED = 15
MAX_PATCHES = 750
PATCH_SIZE = (64, 64)

DATA_FILE = "renoir-Mi3-aligned-64x64-CLEAN-NOISY.h5"


def load_batches(path):
    # Get batches as list of folders
    batches = [f for f in os.listdir(path) if f.startswith("Batch")]
    batches = list(map(lambda x: {"folder": os.path.join(path, x), "noisy": [], "clean": ""}, batches))
    for batch in batches:
        for image in os.listdir(batch["folder"]):
            if image.endswith("Noisy.bmp"):
                batch["noisy"].append(image)
            elif image.endswith("full.bmp"):
                batch["clean"] = image
    return batches


def extract_patches_for_batch(batch):
    # Load clean + NORMALIZE
    clean = cv2.imread(os.path.join(batch["folder"], batch["clean"]))
    clean_norm = clean.astype("float32") / 255.0

    # Extract clean patches from image
    clean_patches = image.extract_patches_2d(clean_norm, PATCH_SIZE, MAX_PATCHES, RANDOM_SEED)

    for noisy_file in batch["noisy"]:
        # Load noisy + NORMALIZE
        noisy = cv2.imread(os.path.join(batch["folder"], noisy_file))
        noisy_norm = noisy.astype("float32") / 255.0

        # Extract noisy patches from image
        noisy_patches = image.extract_patches_2d(noisy_norm, PATCH_SIZE, MAX_PATCHES, RANDOM_SEED)

        print(batch["folder"], "-- DONE -> shape CLEAN | NOISY", clean_patches.shape, noisy_patches.shape)

        return np.array([np.array(clean_patches), np.array(noisy_patches)])


def plot_n_images(images):
    # Plot sample patches
    fig = plt.figure(figsize=(9, 2))
    print(images.shape)
    for index in range(len(images)):
        # display Clean
        cx = plt.subplot(2, 10, index + 1)
        plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)
    plt.show()


def generate_data_from_images_loop():
    # Load batches
    batches = load_batches(filepath)

    # Generate data
    print("Loading data ....")
    data = np.concatenate(np.array([np.array(extract_patches_for_batch(batch)) for batch in batches]), axis=1)
    data[0], data[1] = sklearn.utils.shuffle(data[0], data[1], random_state=RANDOM_SEED)
    print("Data loaded and shuffled", data.shape)

    # Save data to file
    train, validation, test = 0.65, 0.2, 0.15

    split_data = np.split(data, [int(train * len(data[0])), int((train + validation) * len(data[0]))], axis=1)
    print("Saving data as train 0.65, validation 0.2 and test 0.15 ...")
    output_file = h5py.File(DATA_FILE, "w")
    output_file.create_dataset('train_data', data=split_data[0])
    output_file.create_dataset('validation_data', data=split_data[1])
    output_file.create_dataset('test_data', data=split_data[2])
    output_file.close()
    print("File saved.")


def load_data_from_file(pathname):
    print("Loading data from file ...")
    input_file = h5py.File(pathname, "r")
    train_data = input_file["train_data"][:]
    validation_data = input_file["validation_data"][:]
    print("Loaded: Train |", train_data.shape, "Validation |", validation_data.shape)

    plot_n_images(train_data[0][50:60])
    plot_n_images(train_data[1][50:60])


# generate_data_from_images_loop()
load_data_from_file(DATA_FILE)
