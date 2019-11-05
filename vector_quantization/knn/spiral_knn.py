import numpy as np

from vector_quantization.knn.k_nearest_neighbour import KNearestNeighbourClassifier

# KNN parameters
K = 5
MESH_GRID_SIZE = .1
TEST_SIZE = .2

# Data parameters
input_data = '..\\spiral.csv'
csv_separator = ';'
csv_line_separator = '\n'


def load_data():
    """
    load and transform the data
    :return: list of tupels: (features, label) | exmaple: ([0.906112, 0.406602], '1')
    """
    lines = [line.split(csv_separator) for idx, line in
             enumerate(open(input_data, 'r').read().split(csv_line_separator)) if line != '']
    return [([float(e) for e in line[:-1]], int(line[-1])) for line in lines]


if __name__ == '__main__':
    data = load_data()
    knn_classifier = KNearestNeighbourClassifier(K)
    train_data, test_data = knn_classifier.train_test_split(data, TEST_SIZE)
    knn_classifier.fit(data)
    accuracy_score = knn_classifier.accuracy(test_data)
    knn_classifier.plot(MESH_GRID_SIZE, round(accuracy_score * 100))
