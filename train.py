from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOAgent
from config import (
    STATE_DIM,
    ACTION_DIM,
    NUM_EPISODES,
    TRAIN_DOMAIN,
    MODEL_PATH
)
import torch
import os


def load_domain_corpus(domain):
    path = f"data/{domain}/corpus.txt"
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def train():
    corpus = load_domain_corpus(TRAIN_DOMAIN)

    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)

    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    for episode in range(NUM_EPISODES):
        state = env.reset("treatment for diabetes")
        done = False
        trajectory = []

        while not done:
            action_fn, action_idx, logp = agent.select_action(state["vector"])
            next_state, reward, done, _ = env.step(state, action_fn)

            trajectory.append((
                state["vector"],
                action_idx,
                logp,
                reward,
                next_state["vector"]
            ))

            state = next_state

        agent.update(trajectory)

        total_reward = sum(t[3] for t in trajectory)
        print(f"Episode {episode:02d} | Total reward: {total_reward:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), MODEL_PATH)
    print(f"\n✅ Trained PPO policy saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
