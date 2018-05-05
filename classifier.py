"""
Module for training and predicting lung nodule classifier.

ECE 4250 Final Project.
Brian Richard (bcr53), Gautam Mekkat (gm484), Sameer Lal (sjl328).
May 3rd, 2018.
"""

import sklearn.neighbors
import sklearn.externals
import util


def generate_training_output(candidate_nodules, training_nodules):
    """Generate input to the classifier using the generated candidate nodules and training nodules."""
    y = []
    for candidate in candidate_nodules:
        output = 0
        for training in training_nodules:
            if util.distance(candidate[:3], training) < 3:
                output = 1
                break
        y.append(output)
    print(len(candidate_nodules), len(y))
    return y


def classifier(input, output):
    """Train and return classifier using KNN."""
    print(len(input), len(output))
    with open('training.csv', 'w') as f:
        for i in range(len(input)):
            line = ', '.join([str(a) for a in input[i]]) + ', ' + str(output[i])
            f.write(line)
            f.write('\n')
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(input, output)
    sklearn.externals.joblib.dump(knn, 'classifier.pkl')
