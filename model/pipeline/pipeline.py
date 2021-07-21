import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import notebook
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print("started pipeline")


def process_data():
    print("process_data")
    input_images = []
    mask_images = []
    image_path = "../data/CameraRGB/"
    mask_path = "../data/CameraSeg/"
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    image_list = [image_path + i for i in image_list]
    mask_list = [mask_path + i for i in mask_list]

    for i in notebook.tqdm(range(len(image_list))):
        img, mask = cv2.imread(image_list[i]), cv2.imread(mask_list[i])[:, :, 2]
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256,))
        input_images.append(img)
        mask_images.append(mask)

    input_images = np.array(input_images)
    input_images = input_images / 255.
    mask_images = np.array(mask_images)
    mask_images = mask_images.reshape((mask_images.shape[0], mask_images.shape[1], mask_images.shape[2], 1))
    input_img = input_images[0]
    h = input_img.shape[0]
    w = input_img.shape[1]
    l = input_img.shape[2]
    return input_images, mask_images, h, w, l


if __name__ == "__main__":
    process_data()
