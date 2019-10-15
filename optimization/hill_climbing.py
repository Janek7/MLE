

class HillClimber:

    def __init__(self, fitness, move_one_step_at_random, build_start_hypothesis):
        """
        creates an hill climber algorithm
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

    def run(self, data, iterations):
        """
        runs the hill climbing algorithm
        :return: optimal hypothesis  and number of iterations
        """
        self.data = data
        print('-' * 30 + 'Hill Climbing' + '-' * 30)
        hypothesis = self.build_start_hypothesis(data)
        print('Starting hypothesis:', str(hypothesis))
        last_fitness = self.fitness(hypothesis, data)
        print('Starting fitness:', last_fitness)
        print('-' * 40)
        saved_hypothesis = None

        for i in range(1, iterations):

            saved_hypothesis = hypothesis.copy()
            hypothesis = self.move_one_step_at_random(hypothesis)
            if self.fitness(hypothesis, data) > last_fitness:
                last_fitness = self.fitness(hypothesis, data)
                print('New fitness:', last_fitness)
            else:
                hypothesis = saved_hypothesis

        self.optimal_hypothesis = hypothesis
        self.iterations_needed = iterations
        return hypothesis, iterations

    def print_result(self):
        print('-' * 20 + 'Hill Climbing' + '-' * 20)
        print('Optimal fitness:', self.fitness(self.optimal_hypothesis, self.data))
        print('Iterations:', self.iterations_needed)
        print('Shortest round trip:', str(self.optimal_hypothesis),
              'distance:', self.fitness(self.optimal_hypothesis, self.data) * -1)
