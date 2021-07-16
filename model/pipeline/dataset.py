import tensorflow as tf
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from mask_generator import generate_mask

np.random.seed(42)
tf.random.set_seed(42)
H = 256
W = 256


def process_data():
    image_path = "../data/CameraRGB/"
    mask_path = "../data/CameraSeg/"
    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)
    image_list = [image_path + i for i in image_list]
    mask_list = [mask_path + i for i in mask_list]

    new_masks = []
    for img in mask_list:
        m_img = cv2.imread(img)
        m_img = generate_mask(m_img)
        new_masks.append(m_img)

    return image_list, new_masks


def load_data():
    train_x, train_y = process_data()
    test_x, test_y = process_data()

    train_x, valid_x = train_test_split(train_x, test_size=0.3, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.3, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x - 1
    x = x.astype(np.int32)
    return x


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 3])

    return image, mask


if __name__ == "__main__":
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
    dataset = tf_dataset(train_x, train_y, batch=8)

