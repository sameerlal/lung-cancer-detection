"""
Module for generating lung masks.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import numpy as np
import skimage.filters
import skimage.morphology

import util
import plot

LEFT_LUNG_GUESS = [0.6, 0.5, 0.25]  # (z, y, x)
RIGHT_LUNG_GUESS = [0.6, 0.5, 0.75]


def get_lung_mask(img_arr_3d):
    """
    Return an image mask that only highlights the lungs in the given image.

    :param img_arr_3d: 3D image array to mask as numpy array.
    :return: Numpy array of 1s and 0s. 1 means that a lung is at the corresponding location in the given image.
    """
    img_arr = img_arr_3d
    img_arr = util.standardize_and_remove_bg(img_arr)
    rescaled = util.rescale(img_arr, min=0, max=255).astype('uint8')

    # Use Otsu's method to get black and white image (differentiates lung and bone).
    threshold = skimage.filters.threshold_otsu(rescaled)
    binary = rescaled < threshold

    # Morphological opening to get rid of graininess
    mask = skimage.morphology.binary_opening(binary, skimage.morphology.ball(2))

    # Morphological closing to get rid of some black specks in lung
    mask = skimage.morphology.binary_closing(mask, skimage.morphology.ball(5))

    # Connected threshold to get only lungs and not the background
    seed = np.zeros(mask.shape)
    seed_radius = 4
    left_lung_position = tuple(int(mask.shape[i] * LEFT_LUNG_GUESS[i]) for i in range(len(mask.shape)))
    right_lung_position = tuple(int(mask.shape[i] * RIGHT_LUNG_GUESS[i]) for i in range(len(mask.shape)))
    for seed_coord in [left_lung_position, right_lung_position]:
        z1 = seed_coord[0] - seed_radius
        y1 = seed_coord[1] - seed_radius
        x1 = seed_coord[2] - seed_radius

        z2 = seed_coord[0] + seed_radius
        y2 = seed_coord[1] + seed_radius
        x2 = seed_coord[2] + seed_radius
        seed[z1:z2, y1:y2, x1:x2] = mask[z1:z2, y1:y2, x1:x2]
        print(seed_coord, 1 in seed[z1:z2, y1:y2, x1:x2])
    mask = skimage.morphology.reconstruction(seed, mask)

    # Morphological closing to get rid of all black specks in lung
    mask = skimage.morphology.binary_closing(mask, skimage.morphology.ball(7))

    # Dilate by 1 to get edges of lung
    mask = skimage.morphology.binary_dilation(mask, skimage.morphology.ball(1))
    return mask
