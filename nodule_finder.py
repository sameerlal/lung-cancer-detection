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
