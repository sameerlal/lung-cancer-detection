"""
Module for predicting lung scan probabilities using trained classifier.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
May 7th, 2018.
"""

import os
import sys
import numpy as np
from keras.models import load_model

import mask_gen
import nodule_finder
import load_image
import util
import classifier


def predict(model, scan_directory):
    """
    Predict the probability that candidate nodules are suspicious in the given lung scan.

    :param model: The trained classifier.
    :param scan_directory: Name of directory where scans to predict are.
    :return: A list of candidate nodules and the expected probability.
    """
    print('Predicting now')
    open('predictions.csv', 'w').close()  # Clear predictions.csv
    files = load_image.get_files(scan_directory)
    for f in files:
        label = os.path.splitext(os.path.basename(f))[0]
        im = load_image.load_image(f)
        lung_scan = util.get_image_array(im)
        origin = load_image.get_origin(im)
        print(label)
        print('Preprocessing')
        mask = mask_gen.get_lung_mask(lung_scan)
        candidate_nodules = nodule_finder.extract_candidate_nodules_3d(lung_scan, mask)
        x = classifier.generate_testing_input(candidate_nodules)
        prob = []
        for nodule in x:
            prediction = model.predict(np.asarray([util.pad_3d(nodule, 32, 32, 32)]))
            prob.append(prediction)
        with open('predictions.csv', 'a') as f:
            for i in range(len(candidate_nodules)):
                candidate = candidate_nodules[i]
                f.write(label)
                f.write(',')
                true_center = np.array(candidate['center']) + origin
                true_center = true_center.astype(str)
                f.write(','.join(true_center))
                f.write(',')
                f.write(str(prob[i][0]))
                f.write('\n')


if __name__ == '__main__':
    classifier_filename = 'classifier.h5'
    model = load_model(classifier_filename)
    directory = sys.argv[1]
    print('Using scans in directory', directory, 'for testing.')
    predict(model, directory)
