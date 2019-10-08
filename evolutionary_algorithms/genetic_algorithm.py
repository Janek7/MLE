import random
import matplotlib.pyplot as plt

P = 100  # Population size
R = 0.5  # share to replace with crossover
M = 0.1  # share to mutate with one bit
FITNESS_THRESHOLD = 70
STRING_LENGTH = 100
STRING_CONTENT = '1'  # random, 0 or 1


class GeneticAlgorithm:  # hypothesis == individual

    def __init__(self):
        self.optimum_bit_string = self.generate_random_bit_string(STRING_CONTENT)
        print('Optimum bit string:', self.optimum_bit_string)
        # create first generation
        self.population = [self.generate_random_bit_string('random') for individual in range(P)]
        self.generation_counter = 1
        self.fitness_dict = None
        self.update_fitness_dict()
        self.best_hypothesis = None
        self.best_fitness_generation_list = []
        self.update_best_hypothesis()

    def run(self):
        self.print_generation()

        # while self.optimum_bit_string not in self.population:
        while self.fitness_dict[self.best_hypothesis] < FITNESS_THRESHOLD or self.optimum_bit_string in self.population:
            new_generation = []

            # Selection
            while len(new_generation) < P * R:
                new_generation.append(self.select_random_hypothesis(self.population))

            # Crossover
            parents = []
            while len(parents) < R * (P / 2):  # select parents
                # parents aus den bereits ausgewählten (new_generation) oder allen (population)?
                father = self.select_random_hypothesis(self.population)  # <- ERROR
                # mother = new_generation[random.randint(0, len(new_generation) - 1)]
                mother = self.select_random_hypothesis(self.population)
                if father != mother:
                    parents.append((father, mother))
            for parent_tupel in parents:
                new_generation += self.cross_over(parent_tupel)

            # Mutation
            for i in range(int(P * M)):
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

    def fitness(self, bit_string):
        """
        returns the hamming distance between the given string and the optimum_string as the fitness
        :return: fitness
        """
        # return int(len(self.optimum_bit_string) - scipy.spatial.distance.hamming(self.optimum_bit_string, bit_string))
        distance = 0
        for i in range(STRING_LENGTH - 1):
            if bit_string[i] != self.optimum_bit_string[i]:
                distance += 1
        return len(self.optimum_bit_string) - distance

    def select_random_hypothesis(self, population):
        """
        selects a random hypothesis based on the probability of fitness / sum(all fitness)
        :param population: population to select from
        :return: random hypothesis
        """
        selection_probability = random.random()
        probability_sum = 0
        index = random.randint(0, len(population) - 1)
        population_fitness_sum = sum([self.fitness(h) for h in population])

        while True:  # do while loop
            index += 1
            index = index % P
            probability_sum += (self.fitness(population[index]) / population_fitness_sum)
            if selection_probability < probability_sum:  # do while loop
                break

        return population[index]

    @staticmethod
    def cross_over(parents):
        """
        creates two children with cross over from parents based on a random cross over point
        :param parents: parent tupel
        :return: child list
        """
        cross_over_point = random.randint(1, STRING_LENGTH - 1)
        child_one = parents[0][:cross_over_point] + parents[1][cross_over_point:]
        child_two = parents[1][:cross_over_point] + parents[0][cross_over_point:]
        return [child_one, child_two]

    @staticmethod
    def mutate(hypothesis):
        """
        mutates one random index to the complementary bit
        """
        index = random.randint(0, len(hypothesis) - 1)
        return hypothesis[:index] + ('0' if hypothesis[index] == '1' else '1') + hypothesis[index + 1:]

    def update_fitness_dict(self):
        """
        updates self.fitness_dict with individuals of self.population as key and their fitness as value
        """
        self.fitness_dict = {i: self.fitness(i) for i in self.population}

    def update_best_hypothesis(self):
        """
        updates self.best_hypothesis with best key of actual self.fitness_dict
        """
        self.best_hypothesis = max(self.fitness_dict, key=self.fitness_dict.get)
        self.best_fitness_generation_list.append(self.fitness_dict[self.best_hypothesis])

    @staticmethod
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

    def print_generation(self):
        """
        prints information about the actual generation
        """
        print(str('-' * 10 + ' {}. generation ' + '-' * 10).format(self.generation_counter))
        print('Population size: {}'.format(len(self.population)))
        print('Best hypothesis: {}, fitness: {}'.format(self.best_hypothesis, self.fitness(self.best_hypothesis)))

    def print_finish(self):
        """
        prints information after finish
        """
        print('-' * 150)
        print('Target reachted after {} generations'.format(self.generation_counter))
        print('Best hypothesis: {}, fitness: {}'.format(self.best_hypothesis, self.fitness(self.best_hypothesis)))

        plt.plot(range(len(self.best_fitness_generation_list)), self.best_fitness_generation_list)
        plt.show()


if __name__ == '__main__':
    GeneticAlgorithm().run()
