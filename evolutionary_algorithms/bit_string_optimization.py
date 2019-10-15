import random
from evolutionary_algorithms.genetic_algorithm import GeneticAlgorithm

# Parameters
P = 100  # Population size
R = 0.5  # share to replace with crossover
M = 0.1  # share to mutate with one bit
FITNESS_THRESHOLD = 100
STRING_LENGTH = 100
STRING_CONTENT = '1'  # random, 0 or 1


def fitness(bit_string, optimum):
    """
    returns the hamming distance between the given string and the optimum_string as the fitness
    :param bit_string: bit string
    :param optimum: optimal bit string to compare with
    :return: fitness
    """
    distance = 0
    for i in range(STRING_LENGTH - 1):
        if bit_string[i] != optimum[i]:
            distance += 1
    return len(optimum) - distance


def cross_over(parents):
    """
    creates two children with cross over from parents based on a random cross over point
    :param parents: parent tupel
    :return: child list
    """
    cross_over_point = random.randint(1, len(parents[0]) - 1)
    child_one = parents[0][:cross_over_point] + parents[1][cross_over_point:]
    child_two = parents[1][:cross_over_point] + parents[0][cross_over_point:]
    return [child_one, child_two]


def mutate(hypothesis):
    """
    mutates one random index to the complementary bit
    """
    index = random.randint(0, len(hypothesis) - 1)
    return hypothesis[:index] + ('0' if hypothesis[index] == '1' else '1') + hypothesis[index + 1:]


def generate_random_bit_string(string_content):
    """
    generates a random bit string with top parameters
    :param string_content: bit
    :return: generated string
    """
    string = ''
    for i in range(STRING_LENGTH):
        if string_content is 'random':
            string += str(1) if random.random() < 0.5 else str(0)
        else:
            string += str(string_content)

    return string


if __name__ == '__main__':
    population = [generate_random_bit_string('random') for individual in range(P)]
    optimum = generate_random_bit_string('1')
    GeneticAlgorithm(population, fitness, cross_over, mutate, FITNESS_THRESHOLD, R, M, optimum).run()
