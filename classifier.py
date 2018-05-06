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
    found_training = set()
    for candidate in candidate_nodules:
        output = 0
        for training in training_nodules:
            if util.distance(candidate[:3], training[:3]) < 3:
                found_training.add(tuple(training))
                output = 1
                break
        y.append(output)
    print(len(candidate_nodules), len(y))
    not_found = [nodule for nodule in training_nodules if tuple(nodule) not in found_training]
    if len(not_found) > 0:
        for nodule in not_found:
            print('Not found:', nodule)
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
    try:
        knn = sklearn.externals.joblib.load('classifier.pkl')
    except (OSError, IOError) as e:
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(input, output)
    sklearn.externals.joblib.dump(knn, 'classifier.pkl')
