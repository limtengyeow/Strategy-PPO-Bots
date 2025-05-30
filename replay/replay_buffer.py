import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = []
        self.position = 0

    def push(self, episode):
        # episode: list of (state, action, reward, next_state, done, hidden)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        episodes = random.sample(self.buffer, batch_size)
        batch = []
        for episode in episodes:
            # Randomly select a starting point in the episode
            if len(episode) < self.seq_len:
                continue  # Skip short episodes
            start = random.randint(0, len(episode) - self.seq_len)
            sequence = episode[start : start + self.seq_len]

            # Unpack the sequence into separate lists
            states, actions, rewards, next_states, dones, hiddens = zip(*sequence)
            # Stack into tensors (seq_len, feature_dim)
            states = np.stack(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.stack(next_states)
            dones = np.array(dones)
            # Keep initial hidden state only (tuple of (h, c))
            hidden = hiddens[0]
            batch.append((states, actions, rewards, next_states, dones, hidden))
        return batch

    def __len__(self):
        return len(self.buffer)
