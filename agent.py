import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from actions import ACTIONS


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.actions = ACTIONS

    def select_action(self, state_vec):
        state = torch.FloatTensor(state_vec)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

        return (
            self.actions[action_idx.item()],
            action_idx,
            dist.log_prob(action_idx)
        )

    def update(self, trajectories, gamma=0.99):
        returns = []
        G = 0
        for _, _, r, _, _ in reversed(trajectories):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        for (s, a_idx, logp_old, _, _) , Gt in zip(trajectories, returns):
            state = torch.FloatTensor(s)
            value = self.value(state).squeeze()

            advantage = Gt - value.detach()

            probs = self.policy(state)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(a_idx)

            loss = -logp * advantage + 0.5 * (value - Gt) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
