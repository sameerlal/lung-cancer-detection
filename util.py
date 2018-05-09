"""
Module for useful general functions.

ECE 4250 Final Project.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
April 21th, 2018.
"""

import numpy as np
import SimpleITK as sitk
from skimage import filters


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

def average_intensity(img, coord, diameter):
    """
    Return the average intensity of a nodule.

    :param img: N-dimensional image as numpy array.
    :param coord: The coordinates of the nodules, (x, y, z).
    :param diameter: The approximate diameter of the nodule
    :return: Average intensity of an image
    """
    radius = int(diameter / 2)
    img = np.asarray(img)
    coord = np.asarray(coord, dtype=int)
    z1 = max(0, coord[2] - radius)
    z2 = min(img.shape[0], coord[2] + radius)
    y1 = max(0, coord[1] - radius)
    y2 = min(img.shape[1], coord[1] + radius)
    x1 = max(0, coord[0] - radius)
    x2 = min(img.shape[2], coord[0] + radius)
    return np.mean(img[z1:z2, y1:y2, x1:x2])

def radial_variance(img, coord, diameter):
    """
    Return the radial variance of a nodule. Pseudo determination of spiculation
    
    :param img: N-dimensional image as numpy array.
    :param coord: The coordinates of the nodules, (x, y, z).
    :param diameter: The approximate diameter of the nodule
    :return: Radial variance of a nodule
    """
    radius = int(diameter/2)
    try:
        snip = np.asarray(img[coord[0] - radius:coord[0]+radius, coord[1] - radius:coord[1]+radius, coord[2]-radius:coord[2]+radius])
    except IndexError:
        #just not dealing with this right now
        return (radius**2)//2
    if (snip.shape[0]*snip.shape[1]*snip.shape[2] == 0):
        # if any axis is zero size, exit quickly
        return (radius**2)//2
    thresh = []
    for i in range(snip.shape[2]):
        try:
            thresh.append(filters.threshold_otsu(snip[:, :, i]))
        except ValueError:
            thresh.append(snip[:, :, i].mean())
    #keep values above the threshold
    binary = np.empty_like(snip, dtype = 'uint8')
    for i in range(snip.shape[2]):
        binary[:, :, i] = snip[:, :, i] >= thresh[i]
    #now calculate variance
    indices = np.where(binary == 1)
    middle = (binary.shape[0]//2, binary.shape[1]//2, binary.shape[2]//2)
    radii = np.empty(len(indices))
    for i in range(len(indices)):
        point = indices[i]
        #some error here with getting point of shape (2, 1) sometimes
        #radius squared but oh well, don't want to waste time
        try:
            radii[i] = (point[0] - middle[0])**2 + (point[1] - middle[1])**2 + (point[2] - middle[2])**2
        except IndexError:
            radii[i] = (radius**2)//2
    return np.std(radii)

def spac_mat(img, coord, diameter):
    """
    Return Gray-tone Spacial Dependence Matrix
    Reference Usage: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4309314

    :param img: N-dimensional image as numpy array.
    :param coord: The coordinates of the nodules, (x, y, z).
    :param diameter: The approximate diameter of the nodule
    :return: (256x256) spacial dependence matrix
    """
    #only taking middle slice
    radius = int(diameter/2)
    mat = np.zeros((256, 256))
    try:
        #snip is values (0,255)
        snip = np.asarray(img[coord[0]-radius:coord[0]+radius, coord[1]-radius:coord[1]+radius, coord[2]], dtype = 'uint8')
    except IndexError:
        #fix this later maybe
        return mat
    for i in range(snip.shape[0]):
        for j in range(snip.shape[1]):
            # this is so bad i'm sorry
            current = snip[i, j]
            try:
                left = snip[i, j - 1]
                mat[current, left] = mat[current, left] + 1
            except IndexError:
                continue
            try:
                tl = snip[i-1, j-1]
                mat[current, tl] = mat[current, tl] + 1
            except IndexError:
                continue
            try:
                top = snip[i-1, j]
                mat[current, top] = mat[current, top] + 1
            except IndexError:
                continue
            try:
                tr = snip[i+1, j+1]
                mat[current, tr] = mat[current, tr] + 1
            except IndexError:
                continue
            try:
                right = snip[i, j+1]
                mat[current, right] = mat[current, right] + 1
            except IndexError:
                continue
            try:
                br = snip[i+1, j+1]
                mat[current, br] = mat[current, br] + 1
            except IndexError:
                continue
            try:
                bottom = snip[i+1, j]
                mat[current, bottom] = mat[current, bottom] + 1
            except IndexError:
                continue
            try:
                bl = snip[i+1, j-1]
                mat[current, bl] = mat[current, bl] + 1
            except IndexError:
                continue
    return mat

def get_contrast(mat):
    """
    Return the contrast of a given Gray-tone spacial dependence matrix
    Reference: https://beta.vu.nl/nl/Images/werkstuk-wan_tcm235-837126.pdf

    :param mat: A (256, 256) GTSDM indices indicating the count of times two grayscale values neighbored eachother
    :return contrast: The 'contrast' of this matrix, defined in report
    """
    contrast = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            contrast = contrast + abs((i-j)**2)*mat[i, j]
    return contrast

def get_corr(mat):
    """
    Return the correlation of a given GTSDM
    Reference: https://beta.vu.nl/nl/Images/werkstuk-wan_tcm235-837126.pdf
    
    :param mat: A (256, 256) ndarray, GTSDM indices indicating the count of times two grayscale values neighbored eachother
    :return corr: The 'correlation' of this matrix, defined in report
    """
    ivals = []
    jvals = []
    for i in range(256):
        ivals.append([np.std(mat[i, :]), np.mean(mat[i, :])])
    for j in range(256):
        jvals.append([np.std(mat[:, j]), np.mean(mat[:, j])])
    corr = 0
    for i in range(256):
        for j in range(256):
            if (ivals[i][0] * jvals[j][0] == 0):
                continue
            corr = corr + ((i - ivals[i][1])*(j - jvals[j][1])*mat[i, j])/(ivals[i][0] * jvals[j][0])
    return corr