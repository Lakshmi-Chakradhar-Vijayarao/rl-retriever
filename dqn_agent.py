import random
import numpy as np
from actions import ACTIONS


class DQNAgent:
    """
    Lightweight tabular-style DQN baseline.
    """

    def __init__(self, action_dim, lr=0.1, gamma=0.95):
        self.q_table = {}
        self.actions = ACTIONS
        self.lr = lr
        self.gamma = gamma
        self.action_dim = action_dim

    def _key(self, state):
        return tuple(round(x, 2) for x in state)

    def select_action(self, state, eps=0.2):
        key = self._key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)

        if random.random() < eps:
            idx = random.randint(0, self.action_dim - 1)
        else:
            idx = int(np.argmax(self.q_table[key]))

        return self.actions[idx], idx

    def update(self, state, action_idx, reward, next_state):
        s = self._key(state)
        ns = self._key(next_state)

        if ns not in self.q_table:
            self.q_table[ns] = np.zeros(self.action_dim)

        q_old = self.q_table[s][action_idx]
        q_next = np.max(self.q_table[ns])

        self.q_table[s][action_idx] += self.lr * (
            reward + self.gamma * q_next - q_old
        )
