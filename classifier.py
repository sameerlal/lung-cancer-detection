"""
Module for training and predicting lung nodule classifier.

ECE 4250 Final Project.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
May 3rd, 2018.
"""

import sys

import numpy as np
import sklearn.neighbors
import sklearn.linear_model
import sklearn.externals.joblib
import util


def generate_training_output(candidate_nodules, training_nodules):
    """Generate input to the classifier using the generated candidate nodules and training nodules."""
    y = []
    found_training = set()
    for candidate in candidate_nodules:
        output = 0
        for training in training_nodules:
            if util.distance(candidate[:3], training[:3]) < candidate[3] / 2:
                found_training.add(tuple(training))
                output = 1
                break
        y.append(output)
    print(len(candidate_nodules), len(y))
    not_found = [nodule for nodule in training_nodules if tuple(nodule) not in found_training]
    if len(not_found) > 0:
        for nodule in not_found:
            print('**Not found:', nodule)
    x = candidate_nodules
    x.extend(not_found)
    y.extend([1] * len(not_found))
    return x, y


def classifier(input, output):
    """Train and return classifier using KNN."""
    print(len(input), len(output))
    with open('training.csv', 'w') as f:
        for i in range(len(input)):
            line = ', '.join([str(a) for a in input[i]]) + ', ' + str(output[i])
            f.write(line)
            f.write('\n')
    knn = sklearn.linear_model.LogisticRegression()
    input = np.asarray(input)
    input = input[:, 3:]
    knn.fit(input, output)
    sklearn.externals.joblib.dump(knn, 'classifier.pkl')


def load_data_from_csv(csv_filename):
    """Train the classifier from a csv file."""
    array = np.genfromtxt(csv_filename, delimiter=',')
    input = array[:, :-1]
    output = array[:, -1]
    return input, output


if __name__ == '__main__':
    training_csv = sys.argv[1]
    input, output = load_data_from_csv(training_csv)
    classifier(input, output)
    print('Wrote classifier to classifier.pkl')
