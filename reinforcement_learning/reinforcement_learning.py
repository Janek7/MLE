import random


# gamma / discount factor
# y = 0.9
# alpha / learning rate
# a = 0.01
# epsilon / probability for epsilon greedy action selection
# e = 0.1


MAX_X_BALL = 100
MAX_Y_BALL = 100
MAX_X_BAT = 100
# x_velocity: 0 (nach links) und 1 (nach rechts)
# y_velocity: 0 (nach unten) und 1 (nach oben)


class ReinforcementLearning:

    def __init__(self, reward_function, episodes, gamma, learning_rate, select_action_strategy, epsilon=0.1):
        """
        creates a new reinforcement learning environment
        :param reward_function: reward function
        :param episodes: number of episodes to learn
        :param gamma: discount factor
        :param learning_rate: alpha
        :param select_action_strategy: strategy for action selection: 'greedy', 'e_greedy' or 'softmax'
        :param epsilon: probability for epsilon greedy action selection
        """
        # parameter
        self.reward = reward_function
        self.episodes = episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.select_action_strategy = select_action_strategy
        self.epsilon = epsilon

        # attributes
        self.q_table = []  # list of dicts: {state: state_dict, q_value: value}
        self.state = None

    def learn(self):

        self.initialize_q_values_arbitrarily()  # was muss hier gemacht werden?

        for i in range(self.episodes):
            self.state = None  # Initialize state s -> wie initialisieren?

            while True:  # for each step of episode -> was bedeutet das?
                action = self.select_action()
                reward, future_state = self.perform_action(action)
                self.update_q_table(self.state)  # was übergeben?
                self.state = future_state  # nötig wenn das in q_table mitgespeichert wird?

                if True:  # until s is terminal -> was bedeutet das?
                    break

    def q(self, state, action):  # ergebnis irgendwo speichern?
        """
        q learning function
        :param state: state
        :param action: action: 'left' or 'right'
        :return: ?
        """
        return 0

    def q_learning_update_formula(self, state, action):
        qt = self.q(state, action)
        new_qt = qt  # + self.learning_rate * (reward + self.gamma * max(aller möglichen Aktionen für den nächsten Zustand) - qt)
        return new_qt

    def update_q_table(self, state):

        self.q_table = [
            {
                'action': 'left',
                'q_value': self.q(self.state, 'left')
            },
            {
                'action': 'right',
                'q_value': self.q(self.state, 'right')
            }
        ]

    def perform_action(self, action):
        # TODO: State modifizieren
        # reward auf neuen state bezogen?
        return 'reward', 'future_state'

    def select_action(self):
        """
        selects an action to take
        greedy: best reward
        e-greedy: best reward action with probability 1 - self.e, random action with probability self.e
        softmax: use of weighted probabilities
        :return:
        """
        if self.select_action_strategy == 'greedy':
            return self.greedy_selection()
        elif self.select_action_strategy == 'e_greedy':
            return self.q_table[random.randint(0, len(self.q_table) - 1)] if random.random < self.epsilon \
                else self.greedy_selection()
        elif self.select_action_strategy == 'softmax':
            raise Exception('softmax select action strategy not implemented yet')
        else:
            raise Exception('select action strategy must be greedy, e_greedy or softmax')

    def greedy_selection(self):
        """
        greedy action selection
        :return:
        """
        min_state_action_dict = self.q_table[0]
        for q_value_dict in self.q_table[1:]:
            if abs(q_value_dict['q_value']) < abs(min_state_action_dict['q_value']):
                min_state_action_dict = q_value_dict
        return min_state_action_dict

    def initialize_q_values_arbitrarily(self):
        """
        creates a random state and initializes the q value table
        """
        self.state = {
            'x_bat': random.randint(0, MAX_X_BAT),
            'x_ball': random.randint(0, MAX_X_BALL),
            'y_ball': random.randint(0, MAX_Y_BALL),
            'x_velocity': 0 if random.random() < 0.5 else 1,
            'y_velocity': 0 if random.random() < 0.5 else 1,
        }
        self.update_q_table(self.state)


if __name__ == '__main__':
    pass
