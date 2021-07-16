import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from mask_generator import generate_mask
import random


np.random.seed(42)
tf.random.set_seed(42)
H = 256
W = 256
print("started pipeline")
def process_data():
    print("process_data")
    image_path = "../data/CameraRGB/"
    mask_path = "../data/CameraSeg/"
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    image_list = [image_path + i for i in image_list]
    mask_list = [mask_path + i for i in mask_list]

    new_masks = []

    # print(np.unique(plt.imread(mask_list[0])[:, :, 0] * 255))
    # labels = ['Unlabeled', 'Building', 'Fence', 'Other',
    #           'Pedestrian', 'Pole', 'Roadline', 'Road',
    #           'Sidewalk', 'Vegetation', 'Car', 'Wall',
    #           'Traffic sign']
    #
    # for i in range(13):
    #     mask = plt.imread(mask_list[0]) * 255
    #     mask = np.where(mask == i, 255, 0)
    #     mask = mask[:, :, 0]
    #     plt.title(labels[i])
    #     plt.imshow(mask)
    #     plt.show()

    # for img in mask_list:
    #     mask = np.unique(F.interpolate(input=torch.from_numpy(
    #                            plt.imread(img)[:,:,0]*255).\
    #                     unsqueeze(0),
    #                     size=256,
    #                     mode='nearest'))

    N = 1
    img = cv2.imread(image_list[N])
    mask = cv2.imread(mask_list[N])
    mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0],
                                                                                                           img.shape[1])

    # fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    plt.title("plt")
    plt.imshow(mask, cmap='Paired')
    # road = np.zeros((600, 800))
    # road[np.where(mask == 1)[0], np.where(mask == 1)[1]] = 1
    # road[np.where(mask == 2)[0], np.where(mask == 2)[1]] = 2
    # road[np.where(mask == 3)[0], np.where(mask == 3)[1]] = 3
    # road[np.where(mask == 4)[0], np.where(mask == 4)[1]] = 4
    # road[np.where(mask == 5)[0], np.where(mask == 5)[1]] = 5
    # road[np.where(mask == 6)[0], np.where(mask == 6)[1]] = 6
    # road[np.where(mask == 7)[0], np.where(mask == 7)[1]] = 7
    # road[np.where(mask == 8)[0], np.where(mask == 8)[1]] = 8
    # road[np.where(mask == 9)[0], np.where(mask == 9)[1]] = 9
    # road[np.where(mask == 10)[0], np.where(mask == 10)[1]] = 10
    # road[np.where(mask == 11)[0], np.where(mask == 11)[1]] = 11
    # road[np.where(mask == 12)[0], np.where(mask == 12)[1]] = 12
    # road[np.where(mask == 13)[0], np.where(mask == 13)[1]] = 13
    # plt.imshow(road, cmap="Paired")
    # arr[0].imshow(img)
    # arr[0].set_title('Image')
    # arr[1].imshow(mask, cmap='Paired')
    # arr[1].set_title('Segmentation')
    plt.show()




    # plt.title("Image Mask")

    # for n in range(len(mask_list)):
    #     mask = cv2.imread(mask_list[n])
    #     mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])
    #     new_masks.append(mask)
    #
    return image_list, new_masks


i, j = process_data()
