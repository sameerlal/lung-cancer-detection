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
                [x, y, z, radius]
    """
    standardized = util.standardize(img_arr)
    log_buffer = []
    maxima = []
    sigmas = np.linspace(2, 22, 11)
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        gaussian = skimage.filters.gaussian(standardized, sigma=sigma)
        laplace = skimage.filters.laplace(gaussian)  # Important note: scikit-image returns negative second derivative for Laplacian.
        scale_normalized = laplace * sigma ** 2  # scale normalized
        log_buffer.append(scale_normalized)
        if len(log_buffer) >= 2:
            prev_sigma = sigmas[i - 1]
            target_slice = len(log_buffer) - 2
            peaks = skimage.feature.peak_local_max(np.asarray(log_buffer), min_distance=prev_sigma, threshold_abs=0.2, exclude_border=False)
            peaks = peaks[peaks[:, 0] == target_slice]  # If we have 3 logs, pick index 1. If we have 2 logs (only occurs during the second iteration), pick index 0.
            peaks[:, 0] = 2 * prev_sigma * 3 ** 0.5  # Diameter
            maxima.extend(peaks)
            log_buffer = log_buffer[-2:]
        # TODO get last sigma value as well.
    candidates = []
    # slice_index = 278
    # plt.imshow(img_arr[slice_index])
    for point in maxima:
        if mask[tuple(point[1:].astype(int))]:
            # if abs(point[1] - slice_index) < 2:
                # circle = plt.Circle((point[3], point[2]), point[0] * 3 ** 0.5, color='r', fill=False)
                # plt.gca().add_artist(circle)
            candidate = list(point[::-1])
            candidate.append(util.average_intensity(standardized, point[:0:-1], point[0]))
            candidates.append(candidate)
    print('Total of', len(candidates), 'candidates found.')
    # plt.show()
    return candidates
