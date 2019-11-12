from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KNearestNeighbourClassifier:
    k = None
    X = None
    y = None

    def __init__(self, k):
        """
        creates a new knn model
        :param k: number of neighbours to compare
        """
        self.k = k

    def fit(self, X, y):
        """
        saves the training data
        :param X: vectors
        :param y: labels
        :return:
        """
        self.X = X
        self.y = y

    def predict(self, X):
        """
        predicts a label of a given vector
        :param X: vector
        :return: label
        """
        neighbours = [(idx, self.euclidean_distance(X, X_train), self.y[idx]) for idx, X_train in enumerate(self.X)]
        neighbours.sort(key=lambda x: x[1])
        counter = Counter([neighbour[2] for neighbour in neighbours[:self.k]])
        label = counter.most_common()[0][0]
        return label

    @staticmethod
    def euclidean_distance(vector1, vector2):
        """
        computes the euclidean distance between two vectors
        :param vector1: 1
        :param vector2: 2
        :return:
        """
        return pow(sum([abs(vector1[i] - vector2[i]) for i in range(len(vector1))]), 0.5)

    @staticmethod
    def train_test_split(X, y, test_share):
        """
        splits a given data set into train and test data with share of test data
        :param X: vectors
        :param y: labels
        :param test_share: share of test data
        :return: train data, test data
        """
        length = X.shape[0]
        p = np.random.permutation(length)
        X, y = X[p], y[p]
        test_start_index = int(length - test_share * length - 1)
        return X[:test_start_index], X[test_start_index:], y[:test_start_index], y[test_start_index:]

    def accuracy(self, X_test, y_test):
        """
        computes the classifiers accuracy using predictions of given test data
        :param test_data: test data
        :return: accuracy as float
        """
        return sum([1 if prediction == y_test[idx] else 0
                    for idx, prediction in enumerate([self.predict(x) for x in X_test])]) / X_test.shape[0]

    def plot(self, mesh_grid_size, accuracy_score=None):
        """
        plot the test data
        :param mesh_grid_size:
        :param accuracy_score:
        :return:
        """
        X = self.X
        y = self.y

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
