"""
Module for visualizing candidate and training nodules.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import numpy as np
import skimage.filters
import skimage.feature
import matplotlib.pyplot as plt

import util
import plot


def extract_candidate_nodules_3d(img_arr, mask):
    """
    Extract suspicious nodules from masked image.

    :param img_arr: Image array as 3D numpy array.
    :param mask: Image mask as 3D numpy array (must have same shape as img_arr).
    :return: List of candidate nodules. Each candidate nodule is in the format:
                [centroid (x, y, z), volume, fill_factor]
    """
    standardized = util.standardize(img_arr)
    output = []
    sigmas = np.linspace(3, 30, 20)
    for sigma in sigmas:
        print('Sigma:', sigma)
        gaussian = skimage.filters.gaussian(standardized, sigma=sigma)
        print('\tComputed gaussian')
        laplace = skimage.filters.laplace(gaussian)
        print('\tComputed laplacian')
        scale_normalized = laplace * sigma ** 2  # scale normalized
        print('\t\tNormalized')
        output.append(scale_normalized)
        print('\tAppending to output')
    print('Getting peaks now...')
    maxima = skimage.feature.peak_local_max(-np.asarray(output))
    print('Found', len(maxima), 'peaks.')
    candidates = []
    slice_index = 85
    plt.imshow(img_arr[slice_index])
    for point in maxima:
        if mask[tuple(point[1:])]:
            if abs(point[1] - slice_index) < 2:
                circle = plt.Circle((point[3], point[2]), point[0] * 3 ** 0.5, color='r', fill=False)
                plt.gca().add_artist(circle)
            candidates.append([point[:0:-1], point[0]])
            print([point[:0:-1], point[0]])
    plt.show()
    print('Total of', len(candidates), 'candidates found.')
