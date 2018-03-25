"""
ECE 4250 Final Project, Milestone 1
Brian Richard (bcr53), Sameer Lal (sjl328), Gautam Mekkat (gm44)
March 25th, 2018
"""

"""Module for loading and visualizing .mhd/.raw images."""

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage as nd


def load_image(image_file):
    """
    Load image_file and return Image object.

    :param image_file: The filename of the image to load.
    :return: Loaded SimpleITK Image.
    """
    return sitk.ReadImage(image_file)


def get_image_array(im):
    """
    Convert Image object to image array.

    :param im: SimpleITK Image.
    :return: Image array.
    """
    return sitk.GetArrayFromImage(im)


def get_origin(im):
    """
    Get physical origin of image.

    :param im: SimpleITK Image.
    :return: Physical origin of image as a numpy array.
    """
    return np.array(im.GetOrigin())


def get_spacing(im):
    """
    Get voxel spacing of image.

    :param im: SimpleITK image.
    :return: Voxel spacing of image as a numpy array.
    """
    return np.array(im.GetSpacing())


def get_files(directory):
    """
    Get the list of .mhd files in the given directory.

    :param directory: Directory to search through for .mhd files.
    :return: List of filenames (without directory prefix).
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mhd')]

def resample(im, scaling):
	"""
	Resample the input, scaling the axes by the scaling factor.

	:param im: numpy array of Image generated from a .mhd file.
	:param spacing: array of spacing in z, y, x, dimensions.

	:return: A resampled image, as a numpy array.
	"""
	return nd.zoom(im, scaling)

def normalize(im):
	"""
	Normalize a numpy array, scaling matrix values on the interval [0, 1]

	:param im: a nunmpy array to be normalized
	"""
	im_min = np.min(im)
	im_max = np.max(im)
	return (im  - im_min)/(im_max - im_min)

if __name__ == '__main__':
    directory = 'one_test'
    files = get_files(directory)
    slice_number = 60

    for f in files:
        print(f)
        img = load_image(f)
        img_arr = get_image_array(img)

        #resample the image to appropriate spacing (1mm x 1mm x 1mm)
        #reversing order of image spacing to align with ordering in img_arr
        new_img_arr = resample(img_arr, get_spacing(img)[::-1])

        #normalizing the array for plotting purposes
        #in 3d display, the new elemnts can represent the transparency values
        new_img_arr = normalize(new_img_arr)

        #displaying the specified slice
        plt.imshow(new_img_arr[slice_number])
        plt.show()
        