import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# dictionary Keys
from matplotlib.colors import ListedColormap

KEY_VECTOR = 'vector'
KEY_LABEL = 'label'
KEY_NEIGHBOURS = 'neighbours'
KEY_DISTANCE = 'distance'
KEY_VECTOR_KEY = 'vector_key'


class KNearestNeighbourClassifier:
    vectors = {}
    k = None

    def __init__(self, k):
        """
        creates a new knn model
        :param k: number of neighbours to compare
        :param vector_tupels: list of tupels (vector_as_list, label)
        """
        self.k = k

    def fit(self, vector_tupels):
        for vector_tupel in vector_tupels:
            self.add(vector_tupel[0], vector_tupel[1])

    def predict(self, vector):
        """
        predicts the label of a given vector with majority vote of k nearest neighbours
        :param vector: vector to predict
        :return: predicted label index
        """
        key = self.get_key(vector)
        if key in self.vectors:
            return self.vectors[key][KEY_LABEL]

        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        self.add(vector)
        sorted_k_nearest_neighbours = self.vectors[key][KEY_NEIGHBOURS][:self.k]
        counter = Counter([self.vectors[vector_dict[KEY_VECTOR_KEY]][KEY_LABEL]
                           for vector_dict in sorted_k_nearest_neighbours])
        label = counter.most_common()[0][0]
        self.vectors[key][KEY_LABEL] = label
        return label

    @staticmethod
    def euclidean_distance(vector1, vector2):
        """
        computes the euclidean distance between two vectors
        :param vector1:
        :param vector2:
        :return:
        """
        return pow(sum([abs(vector1[i] - vector2[i]) for i in range(len(vector1))]), 0.5)

    def add(self, vector, label=None):
        """
        computes distances of a new vector to all existing vectors, sort them and add the new vector as an dictionary to
        the internal list
        :param vector: new vector
        :param label: new vector's label
        :return:
        """
        sorting = lambda neighbour_dict: neighbour_dict[KEY_DISTANCE]
        # create new vector as dictionary
        vector_dict = {KEY_VECTOR: vector,
                       KEY_LABEL: label,
                       KEY_NEIGHBOURS: [{KEY_VECTOR_KEY: self.get_key(compare_vector),
                                         KEY_DISTANCE: self.euclidean_distance(vector, self.vectors[compare_vector][
                                             KEY_VECTOR])}
                                        for compare_vector in self.vectors]
                       }
        vector_dict[KEY_NEIGHBOURS].sort(key=sorting)

        # add new vector to the other vector's distance lists
        for compare_vector_key in self.vectors:
            compare_vector_dict = self.vectors[compare_vector_key]
            compare_vector_dict[KEY_NEIGHBOURS].append({KEY_VECTOR_KEY: self.get_key(vector),
                                                        KEY_DISTANCE: self.euclidean_distance(
                                                            compare_vector_dict[KEY_VECTOR], vector)
                                                        })
            compare_vector_dict[KEY_NEIGHBOURS].sort(key=sorting)

        # add new vector to internal list
        self.vectors[self.get_key(vector)] = vector_dict

    @staticmethod
    def get_key(vector):
        """
        creates a string from a vector: [1, 2] -> '12'
        :param vector: vector as list
        :return: vector as string
        """
        return ''.join([str(e) for e in vector])

    @staticmethod
    def train_test_split(data, test_size):
        """
        splits a given data set into train and test data with share of test data
        :param data: data
        :param test_size: share of test data
        :return: train data, test data
        """
        random.shuffle(data)
        test_start_index = int(len(data) - test_size * len(data) - 1)
        return data[:test_start_index], data[test_start_index:]

    def accuracy(self, test_data):
        """
        computes the classifiers accuracy using predictions of given test data
        :param test_data: test data
        :return: accuracy as float
        """
        X = [line[0] for line in test_data]
        y = [line[1] for line in test_data]

        return sum([1 if prediction == y[idx] else 0
                    for idx, prediction in enumerate([self.predict(x) for x in X])]) / len(y)

    def plot(self, mesh_grid_size, accuracy_score=None):
        """
        plot the test data
        :param mesh_grid_size:
        :param accuracy_score:
        :return:
        """
        X = np.array([np.array(self.vectors[vector_key][KEY_VECTOR]) for vector_key in self.vectors])
        y = np.array([self.vectors[vector_key][KEY_LABEL] for vector_key in self.vectors])

        if X.shape[1] != 2:
            raise Exception('Only problems with two dimensions can be plotted')

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_grid_size),
                             np.arange(y_min, y_max, mesh_grid_size))
        ravel = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.predict(mesh_vector.tolist()) for mesh_vector in ravel])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        title = 'knn spiral classification (k = {}, records = {} ' + \
                ('accuracy = {}% ' if accuracy_score is not None else '') + ')'
        plt.title(title.format(self.k, ravel.shape[0] + X.shape[0], accuracy_score))
        plt.savefig("knn_spiral.png")
        plt.show()
