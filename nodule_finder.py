"""
Module for visualizing candidate and training nodules.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import skimage

import util
import plot


def extract_candidate_nodules(img, mask):
    """
    Extract suspicious nodules from masked image.

    :param img: Image array as 2D numpy array.
    :param mask: Image mask as 2D numpy array (must have same shape as img).
    :return: Numpy array displaying candidate nodules.
    """
    masked_img = mask * img
    # masked_img[masked_img == 0] = 1
    # Detect candidates blobs within a certain standard deviation
    candidates = feature.blob_log(masked_img, min_sigma=0.5, max_sigma=2, threshold=0.3)
    # Remove blobs on edges
    border = cv2.dilate(mask, np.ones((20, 20))) - cv2.erode(mask, np.ones((6, 6)))
    candidates = [coord for coord in candidates if border[int(coord[0]), int(coord[1])] != 1]
    plt.imshow(masked_img)
    for coord in candidates:
        x = coord[1]
        y = coord[0]
        variance = coord[2]  # Variance of intensity values
        plt.title('Candidate Nodules')
        # Plot on canvas
        circle = plt.Circle((x, y), 1.414 * variance, color='r', fill=False)
        plt.gca().add_artist(circle)
    plt.show()


def extract_candidate_nodules_3d(img_arr, mask):
    """
    Extract suspicious nodules from masked image.

    :param img_arr: Image array as 3D numpy array.
    :param mask: Image mask as 3D numpy array (must have same shape as img_arr).
    :return: Numpy array displaying candidate nodules.
    """
    img = util.standardize(img_arr)
    masked_img = img * mask
    masked_img[masked_img < 0] = 0
    threshold = skimage.filters.threshold_otsu(masked_img)
    binary = masked_img > threshold
    # Get rid of thin paths
    binary = skimage.morphology.binary_opening(binary, skimage.morphology.ball(1))
    labels = skimage.measure.label(binary)
    label_props = skimage.measure.regionprops(labels)
    label_props = sorted(label_props, key=lambda prop: prop.area)
    for prop in label_props:
        print(prop.label, prop.area, prop.bbox)
        xd = prop.bbox[3] - prop.bbox[0]
        yd = prop.bbox[4] - prop.bbox[1]
        zd = prop.bbox[5] - prop.bbox[2]
        diameter = (xd + yd + zd) / 3
        radius = diameter / 2
        volume = 4 / 3 * np.pi * radius ** 3
        fill_factor = prop.area / volume
        print('\tfill factor:', fill_factor)
        print('\t', prop.centroid, '\n')
        # Get rid of non-spherical components (e.g., blood vessels, noise, etc.)
        if fill_factor < 0.4:
            labels[labels == prop.label] = 0
    img = labels > 0
    new_labels = skimage.measure.label(img)
    print(np.max(new_labels), 'CANDIDATE NODULES FOUND')
    plot.plot_slices(img)
