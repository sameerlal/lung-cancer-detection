"""
Module for loading and visualizing .mhd/.raw images.

ECE 4250 Final Project, Milestone 1.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328)
March 25th, 2018
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
from skimage import feature


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


def standardize(img_arr):
    """
    Standardize a numpy array so that the mean is 0 and std. dev. is 1.

    :param img_arr: The numpy array to be standardized.
    :return: The standardized image as a numpy array.
    """
    return (img_arr - np.mean(img_arr)) / np.std(img_arr)


def normalize(img_arr):
    """
    Normalize a numpy array so that all values are between 0 and 1.

    :param img_arr: The numpy array to be normalized.
    :return: The normalized image as a numpy array.
    """
    return (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))


def get_lung_mask(img_arr):
    """
    Return an image mask that only highlights the lungs in the given image.

    :param img_arr: Image array to mask as numpy array.
    :return: Numpy array of 1s and 0s. 1 means that a lung is at the corresponding location in the given image.
    """
    # Credit for some ideas:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html.
    img = standardize(img_arr)
    img[img < 0] = 0  # Get rid of black background.
    img = (normalize(img) * 255).astype('uint8')
    # Blur to get rid of noise.
    blurred = cv2.GaussianBlur(img, (27, 27), 0)
    # Use Otsu's method to get black and white image (differentiates lung and bone).
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Fill background with 0 (we want to mask it out).
    filled = cv2.floodFill(thresh, None, (0, 0), 0)[1]
    filled = cv2.floodFill(filled, None, (img_arr.shape[0] - 1, img_arr.shape[1] - 1), 0)[1]
    # Dilate to get rid of specks in lung mask.
    dilated = cv2.dilate(filled, np.ones((10, 10)))
    # Erode to tighten border of mask.
    eroded = cv2.erode(dilated, np.ones((7, 7)))
    plt.imshow(eroded)
    plt.show()
    return eroded / 255


def extract_candidate_nodules(img, mask):
    """
    Extract suspicious nodules from masked image.

    :param img: Image array as 2D numpy array.
    :param mask: Image mask as 2D numpy array (must have same shape as img).
    :return: Numpy array displaying candidate nodules.
    """
    masked_img = mask * img
    masked_img[masked_img == 0] = 1
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


if __name__ == '__main__':
    # Training data should be in the Traindata_small/ directory
    directory = 'Traindata_small'
    files = get_files(directory)
    training_nodules = load_nodule_csv('training_nodules.csv')

    for f in files:
        print(f)
        label = os.path.splitext(os.path.basename(f))[0]
        img = load_image(f)

        print('Image Spacing (in mm):', get_spacing(img))
        print('Image Origin (in mm):', get_origin(img))
        print('Voxel Spacing (in mm):', get_slice_spacing(img))
        print('Slice Thickness (in mm):', get_slice_thickness(img))

        # Resample the image to appropriate spacing (1mm x 1mm x 1mm).
        img_arr = get_image_array(resample_image_to_1mm(img))
        img_arr = standardize(img_arr)

        # Display the training nodules and get the z-value to plot
        slice_index = img_arr.shape[0] // 2
        if label in training_nodules:
            nodules = training_nodules[label]
            for nodule in nodules:
                origin = get_origin(img)
                x, y, z = nodule[:3] - origin
                slice_index = int(z)
                plt.imshow(img_arr[slice_index], cmap='gray')
                plt.title('{} (slice index {})'.format(label, slice_index))
                print('NODULE FOUND AT ({}, {})'.format(x, y))
                circle = plt.Circle((x, y), nodule[3] / 2, color='r', fill=False)
                plt.gca().add_artist(circle)
                plt.show()
                mask = get_lung_mask(img_arr[slice_index])
                extract_candidate_nodules(img_arr[slice_index], mask)
        else:
            print('NO NODULE FOUND')
            plt.imshow(img_arr[slice_index], cmap='gray')
            plt.title('{} (slice index {})'.format(label, slice_index))
            plt.show()
            mask = get_lung_mask(img_arr[slice_index])
            extract_candidate_nodules(img_arr[slice_index], mask)
