import random
import numpy as np
from SumTree import SumTree

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, state, action, reward, next_state):
        p = self._get_priority(error)
        self.tree.add(p, state, action, reward, next_state)

    def sample(self, n):
        states = []
        actions = []
        rewards = []
        next_states = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, state, action, reward, next_state) = self.tree.get(s)
            priorities.append(p)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return states, actions, rewards, next_states, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
