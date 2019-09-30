import random
import math


ITERATIONS = 100000
NUMBER_OF_NODES = 8
MAX_DISTANCE = 10


def hill_climb(distance_matrix, iterations, simulated_annealing=False):
    """
    runs the hill climbing algorithm
    :param distance_matrix: distances
    :param iterations: number of iterations
    :param simulated_annealing: flag for extra simulated_annealing check
    :return: optimal hypothesis
    """
    hypothesis = list(range(NUMBER_OF_NODES))
    print('Starting hypothesis:', str(hypothesis))
    last_fitness = fitness(hypothesis, distance_matrix)
    print('Starting fitness:', last_fitness)
    saved_hypothesis = None
    temp = 10000000

    for i in range(iterations):

        saved_hypothesis = hypothesis.copy()
        hypothesis = move_one_step_at_random(hypothesis)

        # normal hill climbing
        if fitness(hypothesis, distance_matrix) > last_fitness:
            last_fitness = fitness(hypothesis, distance_matrix)
            print('New fitness:', last_fitness)
        # simulated annealing
        elif simulated_annealing \
                and random.random() < math.exp((fitness(hypothesis, distance_matrix) - last_fitness) / 1):
            last_fitness = fitness(hypothesis, distance_matrix)
            print('New fitness:', last_fitness, '(simulated annealing)')
            temp = temp/math.log(i + 1)
        else:
            hypothesis = saved_hypothesis

    return hypothesis


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


def generate_distance_matrix(number_of_nodes):
    """
    generates a matrix with random distance between a variable number of nodes
    :param number_of_nodes: number of nodes
    :return: two dimensional distance matrix
    """
    distance_matrix = [[0 for i in range(NUMBER_OF_NODES)] for j in range(NUMBER_OF_NODES)]

    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if i == j:
                distance_matrix[i][j] = 0
            elif i > j and distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = random.randint(1, MAX_DISTANCE)

    return distance_matrix


if __name__ == '__main__':

    distance_matrix = generate_distance_matrix(NUMBER_OF_NODES)

    print('Distance matrix:')
    for distance_array in distance_matrix:
        print(distance_array)

    optimal_hypothesis = hill_climb(distance_matrix, ITERATIONS, simulated_annealing=True)
    print('Shortest round trip:', str(optimal_hypothesis))
    print('Optimal fitness:', fitness(optimal_hypothesis, distance_matrix) * -1)
