"""Module for loading and visualizing .mhd/.raw images."""

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


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


if __name__ == '__main__':
    directory = 'Traindata_small'
    files = get_files(directory)

    for f in files:
        print(f)
        img_arr = get_image_array(load_image(f))
        print(img_arr[0][0][0])
        plt.imshow(img_arr[60])
        plt.show()
