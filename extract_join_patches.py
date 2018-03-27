import numpy as np
from itertools import product
import cv2
from sklearn.feature_extraction import image
from skimage import util


def split_to_patches(array, shape, step):
    return util.view_as_windows(array, shape, step)


def average_aligning(array, image_height, image_width, channels, patch_height, patch_width, patch_sum_ver, patch_sum_hor, stride_ver, stride_hor):
    # Initialize result
    result = np.zeros((image_height, image_width, channels))
    divisor = np.zeros((image_height, image_width, channels))

    # Loop every patch
    for i, j in product(range(patch_sum_ver), range(patch_sum_hor)):
        patch = array[i, j]

        result[(i * stride_ver):(i * stride_ver) + patch_height, (j * stride_hor):(j * stride_hor) + patch_width] += patch
        divisor[(i * stride_ver):(i * stride_ver) + patch_height, (j * stride_hor):(j * stride_hor) + patch_width] += 1

    return result / divisor

def gradient_aligning(array, image_height, image_width, channels, patch_height, patch_width, patch_sum_ver, patch_sum_hor, stride_ver, stride_hor):

    # Initialize result
    result = np.zeros((image_height, image_width, channels))
    patch_mask = np.ones((patch_height, patch_width, channels))
    image_ones = np.ones((patch_height, patch_width, channels))

    # Compute gradient
    gradient = np.linspace(0, 1, stride_hor).reshape((stride_hor, 1))
    gradient = np.repeat(gradient, channels).reshape((stride_hor, channels))

    # Compute masks
    patch_gradient = np.ones((patch_width, channels))
    patch_gradient[:stride_hor] = gradient[:]

    # Compute final masks
    horizontal_mask = patch_gradient.reshape((1, patch_width, channels))
    vertical_mask = patch_gradient.reshape((patch_width, 1, channels))

    horizontal_mask_inv = np.ones((1, patch_width, channels)) - horizontal_mask
    vertical_mask_inv = np.ones((patch_width, 1, channels)) - vertical_mask

    patch_mask *= horizontal_mask
    patch_mask *= vertical_mask
    image_mask = image_ones - patch_mask

    # Loop every patch
    for i, j in product(range(patch_sum_ver), range(patch_sum_hor)):
        patch = array[i, j]

        # Slice of image for current patch
        slice_ver = slice((i * stride_ver), (i * stride_ver) + patch_height)
        slice_hor = slice((j * stride_hor), (j * stride_hor) + patch_width)

        # Apply according to patch position
        if i == 0 and j == 0:
            result[slice_ver, slice_hor] = patch
        elif i == 0:
            result[slice_ver, slice_hor] = patch * horizontal_mask + result[slice_ver, slice_hor] * horizontal_mask_inv
        elif j == 0:
            result[slice_ver, slice_hor] = patch * vertical_mask + result[slice_ver, slice_hor] * vertical_mask_inv
        else:
            result[slice_ver, slice_hor] = patch * patch_mask + result[slice_ver, slice_hor] * image_mask

    return result


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

    return gradient_aligning(array, image_height, image_width, channels, patch_height, patch_width, patch_sum_ver, patch_sum_hor, stride_ver, stride_hor)



def test_extraction_reconstruction():

    # Test the patching and unpatching
    width, height, channels = 16, 12, 3
    patch_size = 64
    step = 32
    filename = "test_images/001.bmp"

    array = np.arange(width*height*channels).reshape((height, width, channels))
    image = cv2.imread(filename)
    height, width = image.shape[:2]
    print("Loaded image | Shape:", image.shape)

    # Resize and normalize
    width = int(width / step) * step
    height = int(height / step) * step
    image = cv2.resize(image, (192, 192))
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


# test_extraction_reconstruction()




