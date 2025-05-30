import random

import torch
import torch.nn as nn
import torch.optim as optim


class R2D2Agent(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=128, lr=1e-4, gamma=0.99):
        super(R2D2Agent, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma

        # LSTM Q-network
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

        # Target network (clone for stability)
        self.target_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.target_fc = nn.Linear(hidden_size, action_size)
        self.update_target_network()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, hidden):
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        q_values = self.fc(out)
        return q_values, hidden

    def target_forward(self, x, hidden):
        out, hidden = self.target_lstm(x, hidden)
        q_values = self.target_fc(out)
        return q_values, hidden

    def act(self, state, hidden, epsilon=0.05):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1), hidden
        # state: (batch=1, seq_len, input_size)
        q_values, hidden = self.forward(state, hidden)
        action = torch.argmax(q_values[0, -1]).item()
        return action, hidden

    def update_target_network(self):
        self.target_lstm.load_state_dict(self.lstm.state_dict())
        self.target_fc.load_state_dict(self.fc.state_dict())

    def compute_loss(self, batch):
        losses = []
        for sample in batch:
            states, actions, rewards, next_states, dones, hidden = sample
            states = states.to(self.fc.weight.device)  # (1, seq_len, input_size)
            next_states = next_states.to(self.fc.weight.device)

            actions = (
                torch.tensor(actions, dtype=torch.long)
                .unsqueeze(0)
                .to(self.fc.weight.device)
            )
            rewards = torch.tensor(rewards, dtype=torch.float32).to(
                self.fc.weight.device
            )
            dones = torch.tensor(dones, dtype=torch.float32).to(self.fc.weight.device)

            # Forward pass
            q_values, _ = self.forward(states, hidden)
            q_a = q_values[:, -1, :].gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                target_q_values, _ = self.target_forward(next_states, hidden)
                target_q_max = target_q_values[:, -1, :].max(dim=1)[0]
                target = rewards + self.gamma * (1 - dones) * target_q_max

            loss = self.loss_fn(q_a, target)
            losses.append(loss)

        loss = torch.stack(losses).mean()
        return loss

    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 40.0)
        self.optimizer.step()
        return loss.item()
