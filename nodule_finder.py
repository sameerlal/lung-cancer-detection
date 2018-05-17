"""
Module for visualizing candidate and training nodules.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import numpy as np
import skimage.filters
import skimage.feature
import scipy.ndimage.filters
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
    # Split blob log into two stages because doing it all in one go is too demanding on memory.
    # Note: By splitting blob log we are losing some accuracy but it's not a big deal.
    maxima = blob_log(standardized, min_sigma=2, max_sigma=10, num_sigma=5, min_distance=1)
    maxima = np.vstack((maxima, blob_log(standardized, min_sigma=12, max_sigma=20, num_sigma=5, min_distance=12)))
    candidates = []
    # slice_index = 90
    # plt.imshow(mask[slice_index] * img_arr[slice_index])
    for point in maxima:
        if mask[tuple(point[1:].astype(int))]:
            # if abs(point[1] - slice_index) < 2:
                # circle = plt.Circle((point[3], point[2]), point[0] // 2, color='r', fill=False)
                # plt.gca().add_artist(circle)
            # candidate = list(point[::-1])
            coord = point[:0:-1]
            radius = point[0] / 2
            box = util.get_bounding_box(img_arr, coord, int(radius))
            # box = box[int(box.shape[0] // 2)]  # Get middle slice
            avg_intensity = util.average_intensity(img_arr, point[:0:-1], point[0])
            candidates.append(dict(center=coord, radius=radius, box=box, intensity=avg_intensity))
    # plt.show()
    return candidates


def blob_log(img_arr, min_sigma=2, max_sigma=20, num_sigma=5, threshold=0.2, min_distance=None):
    """Laplacian of Gaussian to find blobs."""
    log = []
    if min_distance is None:
        min_distance = min_sigma
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    for sigma in sigmas:
        gaussian = skimage.filters.gaussian(img_arr, sigma)
        laplace = skimage.filters.laplace(gaussian)
        scale_normalized = laplace * sigma ** 2  # scale normalized
        log.append(scale_normalized)
    peaks = skimage.feature.peak_local_max(np.asarray(log), threshold_abs=threshold, min_distance=min_distance, exclude_border=False)
    diameters = 2 * sigmas[peaks[:, 0]] * 3 ** 0.5
    peaks = np.asarray(peaks, dtype=float)  # Allow for floating point values
    peaks[:, 0] = diameters
    return peaks


def buffered_log(img_arr):
    """Buffered Laplacian of Gaussian. Allows for large sigma ranges but with loss in accuracy."""
    log_buffer = []
    maxima = []
    sigmas = np.linspace(2, 22, 11)
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        gaussian = skimage.filters.gaussian(img_arr, sigma)
        laplace = skimage.filters.laplace(gaussian)
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
    return maxima
