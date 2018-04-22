"""
Module for generating lung masks.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import SimpleITK as sitk

import util
import plot

LEFT_LUNG_GUESS = [0.3, 0.5, 0.5]  # (x, y, z)
RIGHT_LUNG_GUESS = [0.7, 0.5, 0.5]


def get_lung_mask(img_arr_3d):
    """
    Return an image mask that only highlights the lungs in the given image.

    :param img_arr_3d: 3D image array to mask as numpy array.
    :return: Numpy array of 1s and 0s. 1 means that a lung is at the corresponding location in the given image.
    """
    # Credit for some ideas:
    standardized = util.standardize_and_remove_bg(img_arr_3d)
    rescaled = util.rescale(standardized, min=0, max=255).astype('uint8')

    # Use Otsu's method to get black and white image (differentiates lung and bone).
    im = _otsu_filter(sitk.GetImageFromArray(rescaled))

    # Morphological closing to get rid of black specks in lung
    im = _mc_filter(im, 4)

    # Connected component threshold to get only lungs and not the background
    im = _cc_filter(im)

    # Morphological closing to get rid of black specks in lung
    im = _mc_filter(im, 7)

    lung_mask = util.get_image_array(im)
    plot.plot_slices(img_arr_3d * lung_mask)
    return lung_mask


def _otsu_filter(im):
    """
    Execute Otsu thresholding on the given image.

    :param im: SimpleITK image.
    :return: Otsu-thresholded image.
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    return otsu_filter.Execute(im)


def _cc_filter(im):
    """
    Execute connected threshold filter on the given image.

    :param im: SimpleITK image.
    :return: Filtered image.
    """
    dimensions = im.GetSize()
    left_lung_position = [int(dimensions[i] * LEFT_LUNG_GUESS[i]) for i in range(len(dimensions))]
    right_lung_position = [int(dimensions[i] * RIGHT_LUNG_GUESS[i]) for i in range(len(dimensions))]
    return sitk.ConnectedThreshold(im, seedList=[left_lung_position, right_lung_position], lower=1, upper=1)


def _mc_filter(im, kernel_radius):
    """
    Execute morphological closing on the given image.

    :param im: SimpleITK image.
    :param kernel_radius: The kernel radius.
    :return: Filtered image.
    """
    mc = sitk.BinaryMorphologicalClosingImageFilter()
    mc.SetKernelRadius(kernel_radius)
    return mc.Execute(im)