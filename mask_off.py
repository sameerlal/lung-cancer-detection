"""
Module for generating lung masks.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328)
April 14th, 2018
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import SimpleITK as sitk

def get_mask(img):
    """


    :param img:
    :return im: mask applied to image
    """
    #https://docs.opencv.org/3.3.0/d2/dbd/tutorial_distance_transform.html
    #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials
      # /py_imgproc/py_watershed/py_watershed.html?highlight=watershed

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  #laplacian kernel for edge recognition
    sharp = np.copy(img)  #create a copy of the image
    laplacian = cv2.filter2D(sharp, -1, kernel)
    img.astype('uint32')
    enhanced = sharp - laplacian

    #convert to 8bit integer type
    enhanced.astype('uint8')
    laplacian.astype('uint8')

    #Create binary image of source by thresholding
    bw = (255*(img - np.min(img))/(np.max(img) - np.min(img))).astype('uint8')
    _, bw = cv2.threshold(bw, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #Distance transform
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    dist = (dist - np.min(dist))/(np.max(dist) - np.min(dist))

    #Threshold the distance transform and dilate. Threshold value based on trial and error
    _, thresh = cv2.threshold(dist, 0.45, 1, cv2.THRESH_BINARY)
    peaks = cv2.dilate(thresh, (np.ones((3, 3))).astype('uint8'))
    """
    #Now do voodoo magic
    peaks = peaks.astype('uint8')
    _, contours, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for x in contours:
        x = x.astype('uint32')
    markers = np.zeros(peaks.shape, dtype = 'uint32')

    print(peaks.dtype)
    for i in range(len(contours)):
        markers = cv2.drawContours(markers, contours, contourIdx = i, color = i, thickness = -1)
    
    _, markers = cv2.connectedComponents(np.uint8(peaks))
    markers = markers + 1
    markers = np.int32(markers)

    img = np.uint8(np.resize(img, (img.shape[0], img.shape[1], 3)))

    out = cv2.watershed(img, markers)
    """
    plt.imshow(peaks)
    plt.show()

    return img