import random


def get_random_index_with_probabilities(probabilities):
    """
    returns a random index based on probabilities
    :param probabilities: list of probabilities
    :return: index
    """
    selection_probability = random.random()
    probability_sum = 0
    index = random.randint(0, len(probabilities) - 1)
    population_fitness_sum = sum([probabilities[a] for a in range(len(probabilities))])

    while True:  # do while loop
        index += 1
        index = index % len(probabilities)
        probability_sum += (probabilities[index] / population_fitness_sum)
        if selection_probability < probability_sum:  # do while loop
            break

    return index