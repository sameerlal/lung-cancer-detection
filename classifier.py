"""
Module for training and predicting lung nodule classifier.

ECE 4250 Final Project.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
May 3rd, 2018.
"""

import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout

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
    x = [nodule['box'] for nodule in candidate_nodules]
    # x = [np.append(nodule['center'], [nodule['radius'], nodule['intensity']]) for nodule in candidate_nodules]
    return np.asarray(x)


def classifier(input, output, model=None):
    """Train and return classifier using a neural network."""
    print(len(input), len(output))
    dim = 32
    encoded = [[input[i], output[i]] for i in range(len(input))]
    x = []
    for entry in input:
        x.append(util.pad_3d(entry, dim, dim, dim))
    # TODO check if training.npy exists and if so don't overwrite it.
    np.save('training.npy', encoded)
    if model is None:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(dim, dim, dim)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('softmax'))  # softmax gives probabilities
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(np.asarray(x), np.asarray(output), epochs=10, batch_size=16)
    model.save('classifier.h5')
    return model


def load_data_from_csv(csv_filename):
    """Train the classifier from a csv file."""
    array = np.genfromtxt(csv_filename, delimiter=',')
    input = array[:, :-1]
    output = array[:, -1]
    return input, output


def load_data_from_npy(npy_filename):
    """Train the classifier from a npy file (numpy export)."""
    data = np.load(npy_filename)
    input = [row[0] for row in data]
    output = [row[1] for row in data]
    return np.asarray(input), np.asarray(output)


if __name__ == '__main__':
    training_npy = sys.argv[1]
    input, output = load_data_from_npy(training_npy)
    classifier(input, output)
    print('Wrote classifier to classifier.h5')
