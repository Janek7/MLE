from vector_quantization.knn.k_nearest_neighbour import KNearestNeighbourClassifier

# KNN parameters
K = 3

# Data parameters
input_data = 'spiral.csv'
csv_separator = ';'
csv_line_separator = '\n'


def load_data():
    """
    load and transform the data
    :return: list of tupels: (features, label) | exmaple: ([0.906112, 0.406602], '1')
    """
    lines = [line.split(csv_separator) for idx, line in
             enumerate(open(input_data, 'r').read().split(csv_line_separator)) if line != '']
    return [([float(e) for e in line[:-1]], line[-1]) for line in lines]


if __name__ == '__main__':

    data = load_data()
    knn_classifier = KNearestNeighbourClassifier(K, data)
    new_vector = [0.5, 0.5]
    prediction = knn_classifier.predict(new_vector)
    print(prediction)
