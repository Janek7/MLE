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
    lines = [line.split(csv_separator) for idx, line in
             enumerate(open(input_data, 'r').read().split(csv_line_separator)) if line != '']
    X = np.array([np.array([float(e) for e in line[:-1]]) for line in lines])
    y = np.array([int(line[-1]) for line in lines])
    return X, y


if __name__ == '__main__':
    X, y = load_data()
    knn_classifier = KNearestNeighbourClassifier(K)
    X_train, X_test, y_train, y_test = knn_classifier.train_test_split(X, y, TEST_SIZE)
    knn_classifier.fit(X_train, y_train)
    accuracy_score = knn_classifier.accuracy(X_test, y_test)
    print(accuracy_score)
    #knn_classifier.plot(.002,  round(accuracy_score * 100))
