import os
import cv2

batch = "016"
filepath = "D:\RISKO\SKOLA\V-semester\Bakalarka\Datasets\Renoir\Mi3_Aligned\Mi3_Aligned\Batch_" + batch

# Get the clean and noisy filename
for image in os.listdir(filepath):
    if image.endswith("Noisy.bmp"):

        noisy_path = image
    elif image.endswith("full.bmp"):
        clean_path = image

# Load images
noisy = cv2.imread(os.path.join(filepath, noisy_path))
clean = cv2.imread(os.path.join(filepath, clean_path))

# Crop images
x, y = 200, 100
width, height = 1000, 1000
crop_noisy = noisy[y: y + height, x: x + width]
crop_clean = clean[y: y + height, x: x + width]

# Show Images
cv2.imshow("clean", crop_clean)
cv2.imshow("noisy", crop_noisy)
cv2.waitKey()

# Save images
cv2.imwrite("test_images/001-clean.jpg", crop_clean)
cv2.imwrite("test_images/001-noisy.jpg", crop_noisy)



