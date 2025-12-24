from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOAgent
from train import load_corpus
import torch
import os


def main():
    # Load corpus and initialize components
    corpus = load_corpus("data/corpus.txt")
    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)

    agent = PPOAgent(state_dim=4, action_dim=4)

    # ----------------------------
    # Load trained PPO policy
    # ----------------------------
    model_path = "models/policy.pt"
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
        agent.eval()
        print("✅ Loaded trained PPO policy")
    else:
        print("⚠️ No trained policy found — running with untrained policy")

    # Initial query
    query = "treatment for diabetes"
    state = env.reset(query)

    print("\nInitial Query:", query)

    # Show baseline retrieval
    initial_results = retriever.retrieve(query)
    if initial_results:
        print("Initial Top Result:", initial_results[0][1])

    # Select action using PPO policy
    action_fn, action_idx, _ = agent.select_action(state["vector"])

    # Take one environment step
    next_state, reward, done, info = env.step(state, action_fn)

    print("\nChosen Action:", action_fn.__name__)
    print("Rewritten Query:", info["query"])

    if info["after"]:
        print("Top Result After Rewrite:", info["after"][0][1])

    print("Reward:", round(reward, 3))


if __name__ == "__main__":
    main()
