"""
Module for useful general functions.

ECE 4250 Final Project.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import numpy as np
import SimpleITK as sitk


def get_image_array(im):
    """
    Convert Image object to image array.

    :param im: SimpleITK Image.
    :return: Image array as numpy array.
    """
    return sitk.GetArrayFromImage(im)


def standardize(img_arr):
    """
    Standardize a numpy array so that the mean is 0 and std. dev. is 1.

    :param img_arr: The numpy array to be standardized.
    :return: The standardized image as a numpy array.
    """
    return (img_arr - np.mean(img_arr)) / np.std(img_arr)


def rescale(img_arr, min=0, max=1):
    """
    Rescale a numpy array so that all values are between min and max (default: 0 and 1).

    :param img_arr: The numpy array to be normalized.
    :param min: The new minimum value in the image. Default: 0.
    :param max: The new maximum value in the image. Default: 1.
    :return: The normalized image as a numpy array with values between min and max.
    """
    normalized = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
    return normalized * (max - min) + min


def standardize_and_remove_bg(img_arr):
    """
    Standardize lung image and remove black background.

    :param img_arr: N-dimensional image as numpy array.
    :return: Image with standardized values and background removed.
    """
    standardized = standardize(img_arr)
    standardized[standardized < 0] = 0
    return standardized


def distance(point1, point2):
    """Return the distance between two (x, y, z) points."""
    xd = point1[0] - point2[0]
    yd = point1[1] - point2[1]
    zd = point1[2] - point2[2]
    return (xd ** 2 + yd ** 2 + zd ** 2) ** 0.5
