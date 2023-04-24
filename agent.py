from collections import defaultdict

import numpy as np


class Agent:

    def __init__(self, Q, num_action, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = num_action
        self.eps = 1.0
        self.gamma = 0.85
        self.alpha = 0.1
        self.reward = defaultdict(lambda: np.zeros(self.n_actions))
        self.N = defaultdict(lambda: np.zeros(self.n_actions))
        self.history = []
        self.rewardSum = 0

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if (self.mode == 'test_mode') | (np.random.rand() > self.eps):
            action = np.argmax(self.Q[state])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # epsilon decay
        if done:
            if self.eps > 0.01:
                self.eps -= 0.00002
            if self.mode == "mc_control":
                for s, a, r in self.history:
                    self.Q[s][a] += (self.reward[s][a] - self.Q[s][a]) / self.N[s][a]
                self.reward.clear()
                self.history.clear()
                self.rewardSum = 0
        else:
            if self.mode == "mc_control":
                # self.reward[state][action] = reward + (self.reward[state][action] * self.gamma)
                self.rewardSum = reward + (self.rewardSum * self.gamma)
                self.reward[state][action] += self.rewardSum
                self.N[state][action] += 1
                self.history.append((state, action, reward))

            else:
                self.Q[state][action] += self.alpha * (
                            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

