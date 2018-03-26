import numpy as np
from itertools import product
import cv2
from sklearn.feature_extraction import image
from skimage import util


def split_to_patches(array, shape, step):
    return util.view_as_windows(array, shape, step)


def patch_together(array, image_size=(32, 32)):

    # Input format
    assert len(array.shape) == 5

    # Extract variables
    patch_sum_ver, patch_sum_hor, patch_width, patch_height, channels = array.shape
    image_width, image_height = image_size

    # Compute overlap area and stride between patches
    overlap_ver = (patch_sum_ver * patch_height - image_height) / (patch_sum_ver - 1)
    overlap_hor = (patch_sum_hor * patch_width - image_width) / (patch_sum_hor - 1)

    # Check if proper dimensios
    assert int(overlap_hor) == overlap_hor
    assert int(overlap_ver) == overlap_ver

    # Convert to int
    overlap_ver = int(overlap_ver)
    overlap_hor = int(overlap_hor)

    stride_ver = patch_height - overlap_ver
    stride_hor = patch_width - overlap_hor

    # Initialize result
    result = np.zeros((image_height, image_width, channels))
    divisor = np.zeros((image_height, image_width, channels))
    channel_ones = np.ones((patch_height, patch_width, channels))

    # Loop every patch
    for i, j in product(range(patch_sum_ver), range(patch_sum_hor)):
        patch = patches[i, j]
        result[(i * stride_ver):(i * stride_ver) + patch_height, (j * stride_hor):(j * stride_hor) + patch_width] += patch
        divisor[(i * stride_ver):(i * stride_ver) + patch_height, (j * stride_hor):(j * stride_hor) + patch_width] += 1

    return result / divisor


# Test the patching and unpatching
width, height, channels = 16, 12, 3
patch_size = 64
step = 32
filename = "heart.jpg"

array = np.arange(width*height*channels).reshape((height, width, channels))
image = cv2.imread(filename)
height, width = image.shape[:2]
print("Loaded image | Shape:", image.shape)

# Resize and normalize
width = int(width / step) * step
height = int(height / step) * step
image = cv2.resize(image, (224, 160))
height, width = image.shape[:2]
image = image.astype("float32") / 255
print("Resized image | Shape:", image.shape)

cv2.imshow("sample_images_patched", image)
cv2.waitKey()

patches = split_to_patches(image, shape=(patch_size, patch_size, channels), step=step)
print("Patches | Shape:", patches.shape)

patch_count_ver, patch_count_hor = patches.shape[:2]

# Flatten to 1D array of patches
patches = patches.reshape((patch_count_ver * patch_count_hor, patches.shape[3], patches.shape[4], patches.shape[5]))
print("Patches | Reshape:", patches.shape)

# Predict HERE

# Reshape back to 2D array of patches
patches = patches.reshape((patch_count_ver, patch_count_hor, patches.shape[1], patches.shape[2], patches.shape[3]))
print("Patches Reshape:", patches.shape)

# Reconstruct final image from patches
reconstructed = patch_together(patches, image_size=(width, height))
print("Reconstructed:", reconstructed.shape)

cv2.imshow("sample_images_patched", reconstructed)
cv2.waitKey()







