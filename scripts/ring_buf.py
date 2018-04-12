import random

import numpy as np


class AtariRingBuf:
    def __init__(self, size, action_count):
        self.states = [0] * (size + 1)
        self.actions = np.empty((size + 1, action_count), dtype=np.uint8)
        self.next_states = [0] * (size + 1)
        self.rewards = np.empty((size + 1, 1), dtype=np.uint8)
        self.terminals = np.empty((size + 1, 1), dtype=bool)
        self.start = 0
        self.size = size
        self.end = 0
        self.total = 0
        self.action_count = action_count

    def append(self, state, action, next_state, reward, is_terminal):
        buf_action = np.zeros((1, self.action_count))
        buf_action[0, action] = 1

        self.total = self.total + 1
        if self.total > self.size:
            self.total = self.size
        self.states[self.end] = state
        self.actions[self.end] = buf_action
        self.next_states[self.end] = next_state
        self.rewards[self.end] = reward
        self.terminals[self.end] = is_terminal

        self.end = (self.end + 1) % self.size
        if self.end == self.start:
            self.start = (self.start + 1) % self.size

    def get_batch(self, batch_size):
        indices = random.sample(range(self.total), batch_size)
        return [self.states[i] for i in indices], self.actions[indices], [self.next_states[i] for i in indices], \
               self.rewards[indices], \
               self.terminals[indices]

    def __len__(self):
        if self.end < self.start:
            return self.end + self.size - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
