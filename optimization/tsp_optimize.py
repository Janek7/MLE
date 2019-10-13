from optimization.hill_climbing import *
from optimization.simulated_annealing import *
from optimization.tsp_methods import *

NUMBER_OF_NODES = 8
MAX_DISTANCE = 10
METHOD = 'HC'  # HC or SA

# only for hill climbing
ITERATIONS = 100000
# only for simulated annealing
TEMP = 10
EPSILON = 0.1


if __name__ == '__main__':

    # create data
    distance_matrix = generate_distance_matrix(NUMBER_OF_NODES, MAX_DISTANCE)
    print('Distance matrix:')
    for distance_array in distance_matrix:
        print(distance_array)

    # run algorithms
    hillClimber = HillClimber(fitness, move_one_step_at_random, build_start_hypothesis)
    hillClimber.run(distance_matrix, ITERATIONS)
    simulatedAnnealer = SimulatedAnnealer(fitness, move_one_step_at_random, build_start_hypothesis)
    simulatedAnnealer.run(distance_matrix, TEMP, EPSILON)
    print()

    # print results
    hillClimber.print_result()
    simulatedAnnealer.print_result()



