"""Sameer's attempt at load_image."""

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd


def load_itk(filename):
    """
    Load image_file and return Image object.

    :param image_file: The filename of the image to load.
    :return: Loaded SimpleITK Image.
    """
    return sitk.ReadImage(filename)


def get_image_spacing(itkimg):
    """Get image spacing."""
    spacing_array = np.array(itkimg.GetSpacing())
    return spacing_array


def get_image_origin(itkimg):
    """
    Get physical origin of image.

    :param im: SimpleITK Image.
    :return: Physical origin of image as a numpy array.
    """
    return np.array(itkimg.GetOrigin())


def get_voxel_spacing(itkimg):
    """
    Get voxel spacings of image.

    :param im: SimpleITK image.
    :return: Voxel spacing of image as a numpy array.
    """
    spacing = np.array(itkimg.GetSpacing())
    return spacing[0], spacing[1]


def resample_image(im, spacing):
    """
    Resample the input, scaling the axes by the scaling factor.

    :param im: numpy array of Image generated from a .mhd file.
    :param spacing: array of spacing in z, y, x, dimensions.

    :return: A resampled image, as a numpy array.
    """
    return nd.zoom(im, spacing)


def visualize_slice(np_array):
    """Visualize an image slice."""
    plt.imshow(np_array[120])
    plt.show()


def get_slice_thickness(itkimg):
    """Get slice thickness."""
    return np.array(itkimg.GetSpacing())[2]


def resample(im, spacing):
    """
    Resample the input image to 1mm x 1mm x 1mm spacing.

    :param im: numpy array of Image generated from a .mhd file.
    :param spacing: array of spacing in z, y, x, dimensions.

    :return: A resampled image, as a numpy array.
    """
    return nd.zoom(im, spacing)


def normalize(im):
    """
    Normalize a numpy array.

    :param im: a nunmpy array to be normalized
    """
    im_min = np.min(im)
    im_max = np.max(im)
    return (im - im_min) / (im_max - im_min)


if __name__ == '__main__':
    itk_img = load_itk('Traindata_small/train_1.mhd')
    np_scan = sitk.GetArrayFromImage(itk_img)
    print('Image Spacing:  ', get_image_spacing(itk_img))
    print('Image Origin:  ', get_image_origin(itk_img))
    print('Voxel Spacing:  ', get_voxel_spacing(itk_img))
    print('Slice Thickness:  ', get_slice_thickness(itk_img))
    resampled_img = resample(np_scan, get_image_spacing(itk_img)[::-1])
    normalized_img = normalize(resampled_img)
    # print(normalized_img.shape)
    visualize_slice(normalized_img)
    '''
    with open('normalized_data.txt', 'wb') as outfile:
        for slice_2d in normalized_img:
            np.savetxt(outfile, slice_2d)
    '''
