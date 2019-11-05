from collections import Counter

KEY_VECTOR = 'vector'
KEY_LABEL = 'label'
KEY_NEIGHBOURS = 'neighbours'
KEY_DISTANCE = 'distance'
KEY_VECTOR_KEY = 'vector_key'


class KNearestNeighbourClassifier:
    vectors = {}
    k = None

    def __init__(self, k, vector_tupels):
        """
        creates a new knn model
        :param k: number of neighbours to compare
        :param vector_tupels: list of tupels (vector_as_list, label)
        """
        self.k = k
        for vector_tupel in vector_tupels:
            self.add(vector_tupel[0], vector_tupel[1])

    def predict(self, vector):
        """
        predicts the label of a given vector with majority vote of k nearest neighbours
        :param vector: vector to predict
        :return: predicted label index
        """
        if self.get_key(vector) in self.vectors:
            return self.vectors[self.get_key(vector)][KEY_LABEL]

        self.add(vector)
        sorted_k_nearest_neighbours = self.vectors[self.get_key(vector)][KEY_NEIGHBOURS][:self.k]
        counter = Counter([self.vectors[vector_dict[KEY_VECTOR_KEY]][KEY_LABEL]
                           for vector_dict in sorted_k_nearest_neighbours])
        return counter.most_common()[0][0]

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
