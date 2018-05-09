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
import sklearn.neural_network
import sklearn.externals.joblib

import util


def generate_training_input(candidate_nodules, training_nodules):
    """Generate input to the classifier using the generated candidate nodules and training nodules."""
    y = []
    found_training = []
    for candidate in candidate_nodules:
        output = 0
        for training in training_nodules:
            if util.distance(candidate['center'], training['center']) < training['radius']:
                found_training.append(training)
                output = 1
                break
        y.append(output)
    print(len(candidate_nodules), len(y))
    not_found = [nodule for nodule in training_nodules if nodule not in found_training]
    if len(not_found) > 0:
        for nodule in not_found:
            print('** Not found:', nodule['center'])
    x = generate_testing_input(candidate_nodules)
    x.extend(generate_testing_input(not_found))
    y.extend([1] * len(not_found))
    return x, y


def generate_testing_input(candidate_nodules):
    """Generate input to the classifier using the generated candidate nodules."""
    x = [np.append(nodule['center'], [nodule['radius'], nodule['intensity']]) for nodule in candidate_nodules]
    return x


def classifier(input, output, model=None):
    """Train and return classifier using a neural network."""
    print(len(input), len(output))
    if model is None:
        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10, 10, 4))
    x = []
    with open('training.csv', 'w') as f:
        for i in range(len(input)):
            row = np.array(input[i]).flatten()
            line = ', '.join([str(a) for a in row]) + ', ' + str(output[i])
            x.append(row)
            f.write(line)
            f.write('\n')
    model.fit(x, output)
    sklearn.externals.joblib.dump(model, 'classifier.pkl')
    return model


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
