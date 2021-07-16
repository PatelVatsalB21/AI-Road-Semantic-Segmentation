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
    image_path = "../input/lyft-udacity-challenge/dataA/dataA/CameraRGB/"
    mask_path = "../input/lyft-udacity-challenge/dataA/dataA/CameraSeg/"
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    image_list = [image_path + i for i in image_list]
    mask_list = [mask_path + i for i in mask_list]
    input_images = []
    mask_images = []

    for i in notebook.tqdm(range(1000)):
        input_img = cv2.imread(image_list[i])
        input_img = cv2.resize(input_img, (150, 200))
        input_images.append(input_img)
        mask_img = cv2.imread(mask_list[i])
        mask_img = cv2.resize(mask_img, (144, 192))
        new_img = mask_img[:, :, 2]
        mask_images.append(new_img)

    return input_images, mask_images


def processed_dataset():
    input_images, mask_images = process_data()

    input_images = np.array(input_images)
    input_images = input_images / 255.
    mask_images = np.array(mask_images)
    mask_images = mask_images.reshape((mask_images.shape[0], mask_images.shape[1], mask_images.shape[2], 1))

    print(mask_images[0])
    plt.imshow(mask_images[0][:, :, 0])
    plt.show()

    print(input_images.shape, mask_images.shape)

    input_img = input_images[0]
    output_img = mask_images[0]

    h = input_img.shape[0]
    w = input_img.shape[1]
    l = input_img.shape[2]

    print(h, w, l)

    plt.figure()
    plt.imshow(input_img)

    plt.figure()
    plt.imshow(output_img.reshape(output_img.shape[0], output_img.shape[1]))
    plt.show()

    return input_images, mask_images, h, w, l


if __name__ == "__main__":
    processed_dataset()