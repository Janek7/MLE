import random


def fitness(hypothesis, distance_matrix):
    """
    computes the hypothesis' fitness
    :param hypothesis: hypothesis
    :param distance_matrix: distances
    :return: fitness value
    """
    fitness = sum([distance_matrix[hypothesis[i]][hypothesis[i + 1]]
                   for i in range(len(hypothesis)) if i < len(hypothesis) - 1])
    return fitness * -1


def move_one_step_at_random(hypothesis):
    """
    returns a random hypothesis by swapping two random indices
    :param hypothesis: old hypothesis
    :return: new hypothesis
    """
    try:
        # generate random indices
        first_index = random.randint(0, len(hypothesis) - 1)
        second_index = None
        while second_index is None or second_index == first_index:
            second_index = random.randint(0, len(hypothesis) - 1)

        # swaps indices
        tmp = hypothesis[first_index]
        hypothesis[first_index] = hypothesis[second_index]
        hypothesis[second_index] = tmp
    except IndexError:
        pass

    return hypothesis


def build_start_hypothesis(distance_matrix):
    """
    returns the starting hypothesis of a distance matrix
    :param distance_matrix: distance matrix
    :return: starting hypothesis
    """
    return list(range(len(distance_matrix)))


def generate_distance_matrix(number_of_nodes, max_distance):
    """
    generates a matrix with random distance between a variable number of nodes
    :param number_of_nodes: number of nodes
    :param max_distance: maximal distance between to nodes
    :return: two dimensional distance matrix
    """
    distance_matrix = [[0 for i in range(number_of_nodes)] for j in range(number_of_nodes)]

    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i == j:
                distance_matrix[i][j] = 0
            elif i > j and distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = random.randint(1, max_distance)

    return distance_matrix
