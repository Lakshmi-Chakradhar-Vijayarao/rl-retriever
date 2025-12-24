from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOAgent
import torch
import os


def load_corpus(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def train():
    corpus = load_corpus("data/corpus.txt")
    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)

    agent = PPOAgent(state_dim=4, action_dim=4)

    num_episodes = 30

    for episode in range(num_episodes):
        state = env.reset("treatment for diabetes")
        done = False
        trajectory = []

        while not done:
            action_fn, action_idx, logp = agent.select_action(state["vector"])
            next_state, reward, done, info = env.step(state, action_fn)

            trajectory.append((
                state["vector"],     # s
                action_idx,          # a
                logp,                # log π(a|s)
                reward,              # r
                next_state["vector"] # s'
            ))

            state = next_state

        agent.update(trajectory)

        total_reward = sum(t[3] for t in trajectory)
        print(f"Episode {episode:02d} | Total reward: {total_reward:.3f}")

    # ----------------------------
    # Save trained policy (MLOps-ready)
    # ----------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), "models/policy.pt")
    print("\n✅ Trained PPO policy saved to models/policy.pt")


if __name__ == "__main__":
    train()
