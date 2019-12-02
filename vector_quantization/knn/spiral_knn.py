import numpy as np

from vector_quantization.knn.k_nearest_neighbour import KNearestNeighbourClassifier

# KNN parameters
K = 3
MESH_GRID_SIZE = .1
TEST_SIZE = .05

# Data parameters
input_data = '..\\spiral.csv'
csv_separator = ';'
csv_line_separator = '\n'


def load_data():
    lines = [line.split(csv_separator) for idx, line in
             enumerate(open(input_data, 'r').read().split(csv_line_separator)) if line != '']
    X = np.array([np.array([float(e) for e in line[:-1]]) for line in lines])
    y = np.array([int(line[-1]) for line in lines])
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    knn_classifier = KNearestNeighbourClassifier(K)
    knn_classifier.fit(X, y)
    knn_classifier.plot(MESH_GRID_SIZE,  '-')
