import random
from abc import abstractmethod
import numpy as np


class ReinforcementLearningDomain:
    """
    abstract class with domain specific structures of reinforcement learning
    """

    actions = None
    state_max_dimension_sizes = None
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
        self.state_dimension_sizes = reinforcement_learning_domain.state_max_dimension_sizes
        self.episodes = episodes
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.select_action_strategy = select_action_strategy
        self.epsilon = epsilon

        # attributes
        self.q_table = []  # table of x rows (parameter combinations) with y columns (number of actions) with q values
        self.initial_state_parameters = None  # save copy of given start parameters from game
        self.state_terminated = False  # boolean flag which is set by the game
        self.state = None
        self.state_index = None

    def learn(self):
        """
        learns at the basis of rewards to react on specific states
        :return:
        """
        self.initialize_q_values_arbitrarily()

        for i in range(self.episodes):
            print('{}. episode'.format(i))
            self.update_state(self.initial_state_parameters)

            while True:
                action_index = self.select_action()
                reward, future_state = self.reinforcement_learning_domain.action(action_index)
                self.update_q_table(action_index, reward, future_state)
                self.update_state(future_state)
                if self.state_terminated:
                    break

    def update_q_table(self, action_index, reward, future_state):
        """
        updates the q values for given state
        :param action_index: index of performed action
        :param reward: reward of actual state
        :param future_state: future_state as parameter array
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
        qt += self.learning_rate * (reward
                                    + self.discount_factor
                                    * max([self.q_table[future_state_index, i] for i in range(len(self.actions))])
                                    - qt)
        return qt

    def select_action(self):
        """
        selects an action to take
        greedy: best reward
        e-greedy: best reward action with probability 1 - self.epsilon, random action with probability self.epsilon
        softmax: use of weighted probabilities
        :return:
        """
        if self.select_action_strategy == 'greedy':
            return self.greedy_selection(self.state_index)
        elif self.select_action_strategy == 'e_greedy':
            return random.randint(0, len(self.actions) - 1) if random.random() < self.epsilon \
                else self.greedy_selection(self.state_index)
        elif self.select_action_strategy == 'softmax':
            raise Exception('softmax select action strategy not implemented yet')
        else:
            raise Exception('select action strategy must be greedy, e_greedy or softmax')

    def greedy_selection(self, state_index):
        """
        :param state_index: index of state to select action from
        greedy action selection
        :return: index of best action
        """
        action_rewards = self.q_table[state_index]
        max_reward_action_index = 0

        for idx, value in np.ndenumerate(self.q_table[state_index][1:]):
            if value > action_rewards[max_reward_action_index]:
                max_reward_action_index = idx[0]  # idx in numpy enumeration is tupel (idx,)

        return max_reward_action_index

    def initialize_q_values_arbitrarily(self):
        """
        create the q_table as a table with x rows (number of states / properties combinations) and y columns (actions)
        and initialize the cells with random values
        """
        self.q_table = np.random.rand(np.prod(self.state_dimension_sizes), len(self.actions))

    def init_state(self, state):
        """
        initializes the state with a given parameter set as properties and save the parameters
        :param state: state as array of properties
        :return:
        """
        self.initial_state_parameters = state
        self.update_state(state)

    def get_state_index(self, state_indices):
        """
        returns index of multidimensional state in q_table
        :param state_indices: indices of state parameter values
        :return: state index
        """
        state_index = state_indices[0]
        for dimension_idx, size in enumerate(self.state_dimension_sizes[1:]):
            state_index = state_index * size + state_indices[dimension_idx + 1]
        return state_index

    def update_state(self, state):
        self.state = state
        self.state_index = self.get_state_index(state)

    def terminate(self):
        self.state_terminated = True


if __name__ == '__main__':
    pass
