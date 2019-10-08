import random
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    """
    a generic genetic algorithm to optimize a population
    """

    def __init__(self, population, fitness, cross_over, mutate, fitness_threshold, r, m, optimum=None):
        """
        generates a new genetic algorithm
        :param population: population to optimize
        :param fitness: fitness function
        :param cross_over: cross over function
        :param mutate: mutate function
        :param fitness_threshold: fitness value to end when reached
        :param r: share to replace with cross over
        :param m: share to mutate
        :param optimum: optimum hypothesis
        """
        self.population = population
        # Functions
        self.fitness = fitness
        self.cross_over = cross_over
        self.mutate = mutate
        # Parameter
        self.fitness_threshold = fitness_threshold
        self.r = r
        self.m = m
        self.p = len(self.population)
        self.optimum = optimum
        print('Optimum:', self.optimum)
        # create first generation
        self.generation_counter = 1
        self.fitness_dict = None
        self.update_fitness_dict()
        self.best_hypothesis = None
        self.best_fitness_generation_list = []
        self.update_best_hypothesis()

    def run(self):
        self.print_generation()

        while self.fitness_dict[self.best_hypothesis] < self.fitness_threshold and self.optimum not in self.population:
            new_generation = []

            # Selection
            while len(new_generation) < self.p * self.r:
                new_generation.append(self.select_random_hypothesis(self.population))

            # Crossover
            parent_counter = 0
            while parent_counter < self.r * (self.p / 2):
                father = self.select_random_hypothesis(self.population)
                mother = self.select_random_hypothesis(self.population)
                if father != mother:
                    parent_counter += 1
                    new_generation += self.cross_over((father, mother))

            # Mutation
            for i in range(int(self.p * self.m)):
                index = random.randint(0, len(new_generation) - 1)
                new_generation[index] = self.mutate(new_generation[index])

            # keep best individual
            if self.best_hypothesis not in new_generation:
                new_generation.append(self.best_hypothesis)

            # update population and fitness_dict
            self.population = new_generation
            self.generation_counter += 1
            self.update_fitness_dict()
            self.update_best_hypothesis()
            self.print_generation()

        self.print_finish()

    def select_random_hypothesis(self, population):
        """
        selects a random hypothesis based on the probability of fitness / sum(all fitness)
        :param population: population to select from
        :return: random hypothesis
        """
        selection_probability = random.random()
        probability_sum = 0
        index = random.randint(0, len(population) - 1)
        population_fitness_sum = sum([self.fitness(h, self.optimum) for h in population])

        while True:  # do while loop
            index += 1
            index = index % self.p
            probability_sum += (self.fitness(population[index], self.optimum) / population_fitness_sum)
            if selection_probability < probability_sum:  # do while loop
                break

        return population[index]

    def update_fitness_dict(self):
        """
        updates self.fitness_dict with individuals / hypotheses of self.population as key and their fitness as value
        """
        self.fitness_dict = {h: self.fitness(h, self.optimum) for h in self.population}

    def update_best_hypothesis(self):
        """
        updates self.best_hypothesis with best key of actual self.fitness_dict
        """
        self.best_hypothesis = max(self.fitness_dict, key=self.fitness_dict.get)
        self.best_fitness_generation_list.append(self.fitness_dict[self.best_hypothesis])

    def print_generation(self):
        """
        prints information about the actual generation
        """
        print(str('-' * 10 + ' {}. generation ' + '-' * 10).format(self.generation_counter))
        print('Population size: {}'.format(self.p))
        print('Best hypothesis: {}, fitness: {}'.format(self.best_hypothesis, self.fitness(self.best_hypothesis,
                                                                                           self.optimum)))

    def print_finish(self):
        """
        prints information after finish
        """
        print('-' * 150)
        print('Target reachted after {} generations'.format(self.generation_counter))
        print('Best hypothesis: {}, fitness: {}'.format(self.best_hypothesis, self.fitness(self.best_hypothesis,
                                                                                           self.optimum)))
        plt.plot(range(len(self.best_fitness_generation_list)), self.best_fitness_generation_list)
        plt.show()
