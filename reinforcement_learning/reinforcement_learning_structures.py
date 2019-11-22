import random
from abc import abstractmethod
import numpy as np
import math

from utils import get_random_index_with_probabilities


# selection strategies
GREEDY = 'greedy'
E_GREEDY = 'e_greedy'
SOFTMAX = 'softmax'


class ReinforcementLearningDomain:
    """
    abstract class with domain specific structures of reinforcement learning
    """

    actions = None
    state_dimension_values = None
    agent = None

    @abstractmethod
    def action(self, action_index):
        pass


class ReinforcementLearningAgent:
    """
    reinforcement learning algorithm
    """

    def __init__(self, reinforcement_learning_domain,
                 episodes, discount_factor, learning_rate, select_action_strategy, epsilon=0.1):
        """
        creates a new reinforcement learning environment
        :param reinforcement_learning_domain: application domain with domain specific structures as actions and max
                                              dimension sizes
        :param episodes: number of episodes to learn
        :param discount_factor: discount factor
        :param learning_rate: alpha
        :param select_action_strategy: strategy for action selection: 'greedy', 'e_greedy' or 'softmax'
        :param epsilon: probability for epsilon greedy action selection
        """

        # parameter
        self.reinforcement_learning_domain = reinforcement_learning_domain
        self.actions = reinforcement_learning_domain.actions
        self.state_dimension_values = reinforcement_learning_domain.state_dimension_values
        self.dimension_sizes = [len(dimension_value_list) for dimension_value_list in self.state_dimension_values]
        self.episodes = episodes
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.select_action_strategy = select_action_strategy
        self.epsilon = epsilon

        # attributes
        self.q_table = []  # table of x rows (parameter combinations) with y columns (number of actions) with q values
        self.initial_state = None  # save copy of given start parameters from game
        self.state_terminated = False  # boolean flag which is set by the game
        self.state_index = None

    def learn(self):
        """
        learns at the basis of rewards to react on specific states
        :return:
        """
        self.initialize_q_values_arbitrarily()

        for i in range(self.episodes):
            print('{}. episode'.format(i))
            self.state_terminated = False
            self.update_state_index(self.initial_state)

            while True:
                self.state_reaction()
                if self.state_terminated:
                    break

    def state_reaction(self, state=None):

        state_index = self.state_index if state is None else self.get_state_index(state)

        action_index = self.select_action(state_index)
        reward, future_state = self.reinforcement_learning_domain.action(action_index)
        self.update_q_table(action_index, reward, future_state)
        self.update_state_index(future_state)  # for internal training use
        return action_index  # for external use

    def update_q_table(self, action_index, reward, future_state):
        """
        updates the q values for given state
        :param action_index: index of performed action
        :param reward: reward of actual state
        :param future_state: future_state as dimension value array
        :return:
        """
        future_state_index = self.get_state_index(future_state)
        self.q_table[self.state_index, action_index] \
            = self.q_learning_update_formula(self.state_index, action_index,
                                             reward, future_state_index)

    def q_learning_update_formula(self, state_index, action_index, reward, future_state_index):
        """
        updates a the reward q value of state using reward and future state
        :param state_index: index of actual state
        :param action_index: index of performed action
        :param reward: reward from performed action
        :param future_state_index: index of future state
        :return: new q value for performed action and state
        """
        qt = self.q_table[state_index, action_index]
        qt += self.learning_rate * (reward + self.discount_factor * max(self.q_table[future_state_index]) - qt)
        return qt

    def select_action(self, state_index):
        """
        selects an action to take
        :param state_index: index of current state
        greedy: best reward
        e-greedy: best reward action with probability 1 - self.epsilon, random action with probability self.epsilon
        softmax: use of weighted probabilities
        :return:
        """
        if self.select_action_strategy == GREEDY:
            return self.greedy_selection(state_index)

        elif self.select_action_strategy == E_GREEDY:
            return random.randint(0, len(self.actions) - 1) if random.random() < self.epsilon \
                else self.greedy_selection(state_index)

        elif self.select_action_strategy == SOFTMAX:
            probabilities = [(math.exp(self.q_table[state_index, a] / state_index) /
                              sum([math.exp(self.q_table[state_index, b] / state_index)
                                   for b in range(len(self.actions)) if b != a]))
                             for a in range(len(self.actions))]
            return get_random_index_with_probabilities(probabilities)
        else:
            raise Exception('select action strategy must be greedy, e_greedy or softmax')

    def greedy_selection(self, state_index):
        """
        :param state_index: index of state to select action from
        greedy action selection
        :return: index of best action
        """
        action_rewards = self.q_table[state_index]
        if self.reinforcement_learning_domain.learning is False:
            # print(action_rewards)
            pass
        max_reward_action_index = 0

        for idx, value in np.ndenumerate(self.q_table[state_index]):
            if value > action_rewards[max_reward_action_index]:
                max_reward_action_index = idx[0]  # idx in numpy enumeration is tupel (idx,)

        return max_reward_action_index

    def softmax_e_quotient(self, state_index, action_index):
        return math.exp(self.q_table[state_index, action_index] / state_index)

    def initialize_q_values_arbitrarily(self):
        """
        create the q_table as a table with x rows (number of states / properties combinations) and y columns (actions)
        and initialize cells with random values ((e)greedy selection) or zeros (softmax selection)
        """
        shape = (np.prod([len(single_dimension_values) for single_dimension_values in
                          self.state_dimension_values]), len(self.actions))
        if self.select_action_strategy in [GREEDY, E_GREEDY]:
            self.q_table = np.random.rand(shape[0], shape[1])
        elif self.select_action_strategy == SOFTMAX:
            self.q_table = np.zeros(shape)

    def init_state(self, state):
        """
        initializes the state with a given parameter set as properties and save the parameters
        :param state: state as array of properties
        :return:
        """
        self.initial_state = state
        self.update_state_index(state)

    def get_state_index(self, state):
        """
        returns index of multidimensional state in q_table
        :param state: list of state parameter values
        :return: state index
        """
        dim_value_indices = [np.where(self.state_dimension_values[dim_idx] == dim_value)[0][0]
                             for dim_idx, dim_value in enumerate(state)]

        state_index = dim_value_indices[0]
        for dimension_idx, size in enumerate(self.dimension_sizes[1:]):
            state_index = state_index * size + dim_value_indices[dimension_idx + 1]
        return state_index

    def update_state_index(self, state):
        self.state_index = self.get_state_index(state)
