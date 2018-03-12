import os
import h5py
import cv2


filepath = "D:\RISKO\SKOLA\V-semester\Bakalarka\Datasets\Renoir\Mi3_Aligned\Mi3_Aligned"


def load_batches(path):
    batches = [f for f in os.listdir(path) if f.startswith("Batch")]
    batches = list(map(lambda x: {"folder": os.path.join(path, x), "noisy": [], "clean": ""}, batches))
    for batch in batches:
        for image in os.listdir(batch["folder"]):
            if image.endswith("Noisy.bmp"):
                batch["noisy"].append(image)
            elif image.endswith("full.bmp"):
                batch["clean"] = image
    return batches

def load_images_loop():
    batches = load_batches(filepath)
    for batch in batches:
        pass

