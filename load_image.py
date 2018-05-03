"""
Module for loading and visualizing .mhd/.raw images.

ECE 4250 Final Project, Milestone 1.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
March 25th, 2018.
"""

import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

import util
import mask_gen
import nodule_finder


def load_image(image_file):
    """
    Load image_file and return Image object.

    :param image_file: The filename of the image to load.
    :return: Loaded SimpleITK Image.
    """
    return sitk.ReadImage(image_file)


def load_nodule_csv(csv_file):
    """
    Load csv file and return a dictionary mapping from the first column of each row to the remaining columns.

    :param csv_file: Filename of the csv to load.
    :return: Loaded dictionary.
    """
    output = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            label = line[0]
            coords = np.array([float(i) for i in line[1:]])
            if label not in output:
                output[label] = [coords]
            else:
                output[label].append(coords)
    return output


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


if __name__ == '__main__':
    # Training data should be in the Traindata_small/ directory
    directory = 'Traindata_small'
    files = get_files(directory)
    training_nodules = load_nodule_csv('training_nodules.csv')
    train = sys.argv[1]

    for f in files:
        label = os.path.splitext(os.path.basename(f))[0]
        if label != train:
            continue
        im = load_image(f)

        print('Image Spacing (in mm):', get_spacing(im))
        print('Image Origin (in mm):', get_origin(im))
        print('Voxel Spacing (in mm):', get_slice_spacing(im))
        print('Slice Thickness (in mm):', get_slice_thickness(im))

        # Resample the image to appropriate spacing (1mm x 1mm x 1mm).
        resampled = resample_image_to_1mm(im)  # SimpleITK image
        img_arr = util.get_image_array(resampled)  # Numpy array
        img_arr = util.standardize_and_remove_bg(img_arr)

        # Display the training nodules and get the z-value to plot
        lung_mask = mask_gen.get_lung_mask(img_arr)
        slice_index = img_arr.shape[0] // 2
        nodule_finder.extract_candidate_nodules_3d(img_arr, lung_mask)
        if label in training_nodules:
            nodules = training_nodules[label]
            for nodule in nodules:
                origin = get_origin(im)
                x, y, z = nodule[:3] - origin
                slice_index = int(z)
                plt.imshow(img_arr[slice_index], cmap='gray')
                plt.title('{} (slice index {})'.format(label, slice_index))
                print('NODULE FOUND AT ({}, {})'.format(x, y))
                circle = plt.Circle((x, y), nodule[3] / 2, color='r', fill=False)
                plt.gca().add_artist(circle)
                plt.show()
        else:
            print('NO NODULE FOUND')
            plt.imshow(img_arr[slice_index], cmap='gray')
            plt.title('{} (slice index {})'.format(label, slice_index))
            plt.show()
