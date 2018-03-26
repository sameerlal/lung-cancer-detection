"""
Module for loading and visualizing .mhd/.raw images.

ECE 4250 Final Project, Milestone 1.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328)
March 25th, 2018
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
    :return: Image array as numpy array.
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
    Get voxel spacings of image.

    :param im: SimpleITK image.
    :return: Voxel spacing of image as a numpy array.
    """
    return np.array(im.GetSpacing())


def get_slice_spacing(im):
    """
    Get voxel spacing on axial slice of image.

    :param im: SimpleITK image.
    :return: Voxel spacing of image for an axial slice as a numpy array.
    """
    return get_spacing(im)[:2]


def get_slice_thickness(im):
    """
    Get slice thickness for an axial slice of image.

    :param im:  SimpleITK image.
    :return:  Voxel spacing between axial slices in z-dimension
    """
    return get_spacing(im)[2]


def get_files(directory):
    """
    Get the list of .mhd files in the given directory.

    :param directory: Directory to search through for .mhd files.
    :return: List of filenames (without directory prefix).
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mhd')]


def resample_image_to_1mm(im):
    """
    Resample the input to 1mm x 1mm x 1mm voxel spacing.

    :param im: SimpleITK Image.
    :return: Resampled SimpleITK Image to 1mm x 1mm x 1mm spacing.
    """
    dimensions = np.multiply(im.GetSize(), get_spacing(im))
    dimensions = [int(d + 0.5) for d in dimensions]  # sitk.Image only takes native int type, not numpy types

    # Create reference image on to which we will map the original image.
    # Resource used for resampling:
    # https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/70_Data_Augmentation.ipynb
    reference_image = sitk.Image(*dimensions, im.GetPixelIDValue())
    reference_image.SetOrigin(get_origin(im))
    reference_image.SetDirection(im.GetDirection())
    reference_image.SetSpacing([1, 1, 1])
    reference_image = sitk.Resample(sitk.SmoothingRecursiveGaussian(im, 2), reference_image)
    return sitk.Resample(im, reference_image)


def normalize(im):
    """
    Normalize a numpy array, scaling matrix values on the interval [0, 1].

    :param im: numpy array to be normalized.
    """
    im_min = np.min(im)
    im_max = np.max(im)
    return (im - im_min) / (im_max - im_min)


if __name__ == '__main__':
    # Training data should be in the Traindata_small/ directory
    directory = 'Traindata_small'
    files = get_files(directory)

    for f in files:
        print(f)
        img = load_image(f)

        print('Image Spacing (in mm):', get_spacing(img))
        print('Image Origin (in mm):', get_origin(img))
        print('Voxel Spacing (in mm):', get_slice_spacing(img))
        print('Slice Thickness (in mm):', get_slice_thickness(img))

        # Resample the image to appropriate spacing (1mm x 1mm x 1mm).
        new_img_arr = get_image_array(resample_image_to_1mm(img))

        # Normalize the array for plotting purposes (not used for now).
        # In 3D display, the new elements can represent the transparency values.
        # new_img_arr = normalize(new_img_arr)

        # Display the specified slice (cm.bone is used to show a high constrast image).
        middle_slice_index = new_img_arr.shape[0] // 2
        plt.imshow(cm.bone(new_img_arr[middle_slice_index]))
        plt.show()
