import os
import cv2
from tqdm import notebook
import numpy as np


def process_data():
    image_path1 = "../input/lyft-udacity-challenge/dataA/dataA/CameraRGB/"
    mask_path1 = "../input/lyft-udacity-challenge/dataA/dataA/CameraSeg/"
    image_path2 = "../input/lyft-udacity-challenge/dataB/dataB/CameraRGB/"
    mask_path2 = "../input/lyft-udacity-challenge/dataB/dataB/CameraSeg/"
    image_path3 = "../input/lyft-udacity-challenge/dataC/dataC/CameraRGB/"
    mask_path3 = "../input/lyft-udacity-challenge/dataC/dataC/CameraSeg/"
    image_path4 = "../input/lyft-udacity-challenge/dataD/dataD/CameraRGB/"
    mask_path4 = "../input/lyft-udacity-challenge/dataD/dataD/CameraSeg/"
    image_path5 = "../input/lyft-udacity-challenge/dataE/dataE/CameraRGB/"
    mask_path5 = "../input/lyft-udacity-challenge/dataE/dataE/CameraSeg/"

    list1 = [image_path1, image_path2, image_path3, image_path4, image_path5]
    list2 = [mask_path1, mask_path2, mask_path3, mask_path4, mask_path5]
    image_list = []
    mask_list = []
    input_images = []
    mask_images = []

    for i in list1:
        x = os.listdir(i)
        x = [i + j for j in x]
        image_list = image_list + x

    for i in list1:
        x = os.listdir(i)
        x = [i + j for j in x]
        mask_list = mask_list + x

    for i in notebook.tqdm(range(len(image_list))):
        img = cv2.imread(image_list[i])
        mask = cv2.imread(mask_list[i])[:, :, 2]
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

