import math
import random


class SimulatedAnnealer:

    def __init__(self, fitness, move_one_step_at_random, build_start_hypothesis):
        """
        creates an simulated annealing algorithm
        :param fitness: function
        :param move_one_step_at_random: function
        :param build_start_hypothesis: function
        """
        self.fitness = fitness
        self.move_one_step_at_random = move_one_step_at_random
        self.build_start_hypothesis = build_start_hypothesis
        self.data = None
        self.optimal_hypothesis = None
        self.iterations_needed = None

    def run(self, data, temp, epsilon):
        """
        runs the simulated annealing algorithm
        :param data: data to optimize (f.e. distance matrix)
        :param temp: start temperature
        :param epsilon: value to decrease the temperature in each iteration
        :return: optimal hypothesis and number of iterations
        """
        self.data = data
        print('-' * 30 + 'Simulated Annealing' + '-' * 30)
        hypothesis = self.build_start_hypothesis(data)
        print('Starting hypothesis:', str(hypothesis))
        last_fitness = self.fitness(hypothesis, data)
        print('Starting fitness:', last_fitness)
        print('-' * 40)
        saved_hypothesis = None
        i = 0

        while True:  # do while

            saved_hypothesis = hypothesis.copy()
            hypothesis = self.move_one_step_at_random(hypothesis)
            i += 1

            new_fitness = self.fitness(hypothesis, data)
            if new_fitness > last_fitness:
                last_fitness = new_fitness
            else:
                probability = math.exp((new_fitness - last_fitness) / temp)
                print(probability)
                if random.random() < probability:
                    last_fitness = new_fitness
                    print('New fitness:', last_fitness, '( temp:', temp, ')')
                else:
                    hypothesis = saved_hypothesis
            temp -= epsilon

            if temp < epsilon:  # do while
                print('Temperature:', temp, 'epsilon:', epsilon)
                break

        self.optimal_hypothesis = hypothesis
        self.iterations_needed = i
        return hypothesis, i

    def print_result(self):
        print('-' * 20 + 'Simulated Annealing' + '-' * 20)
        print('Optimal fitness:', self.fitness(self.optimal_hypothesis, self.data))
        print('Iterations:', self.iterations_needed)
        print('Shortest round trip:', str(self.optimal_hypothesis),
              'distance:', self.fitness(self.optimal_hypothesis, self.data) * -1)
