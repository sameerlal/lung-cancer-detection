"""
Module for predicting lung scan probabilities using trained classifier.

ECE 4250 Final Project, Milestone 2.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
May 7th, 2018.
"""

import sys
import sklearn.externals.joblib

import mask_gen
import nodule_finder
import load_image
import classifier


def predict(model):
    """
    Predict the probability that candidate nodules are suspicious in the given lung scan.

    :param model: The trained classifier.
    :return: A list of candidate nodules and the expected probability.
    """
    # print('Preprocessing')
    # mask = mask_gen.get_lung_mask(lung_scan)
    # candidate_nodules = nodule_finder.extract_candidate_nodules_3d(lung_scan, mask)
    print('Predicting now')
    candidate_nodules, expected_output = classifier.load_data_from_csv('training.csv')
    with open('predictions.csv', 'w') as f:
        prob = model.predict_proba(candidate_nodules)
        for i in range(len(candidate_nodules)):
            f.write(','.join([str(n) for n in candidate_nodules[i]]))
            f.write(',')
            f.write(str(expected_output[i]))
            f.write(',')
            f.write(str(prob[i][1]))
            f.write('\n')


if __name__ == '__main__':
    classifier_filename = 'classifier.pkl'
    model = sklearn.externals.joblib.load(classifier_filename)
    predict(model)
