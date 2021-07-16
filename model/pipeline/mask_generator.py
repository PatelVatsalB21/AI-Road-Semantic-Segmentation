import numpy as np


def generate_mask(mask_img):
    new_mask_img = np.zeros((mask_img.shape[0], mask_img.shape[1], 13))

    for j in range(13):
        for k in range(mask_img.shape[0]):
            for l in range(mask_img.shape[1]):
                if mask_img[k, l, 2] == j:
                    new_mask_img[k, l, j] = j
    return new_mask_img
