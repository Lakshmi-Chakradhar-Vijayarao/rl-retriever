import random
from actions import ACTIONS


class PPOQueryRewriteAgent:
    def __init__(self):
        self.actions = ACTIONS
        self.action_probs = [1 / len(ACTIONS)] * len(ACTIONS)
        self.action_history = []

    def select_action(self):
        action = random.choices(
            self.actions,
            weights=self.action_probs,
            k=1
        )[0]

        self.action_history.append(action)
        return action

    def update_policy(self, rewards, lr=0.05):
        """
        Update policy by reinforcing actions that led to positive rewards.
        """

        for action, reward in zip(self.action_history, rewards):
            if reward > 0:
                idx = self.actions.index(action)
                self.action_probs[idx] += lr

        # Normalize probabilities
        total = sum(self.action_probs)
        self.action_probs = [p / total for p in self.action_probs]

        # Clear history after update
        self.action_history = []
