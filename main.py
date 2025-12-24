from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOAgent
from config import STATE_DIM, ACTION_DIM, TRAIN_DOMAIN, MODEL_PATH
import torch
import os


def load_domain_corpus(domain):
    path = f"data/{domain}/corpus.txt"
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    corpus = load_domain_corpus(TRAIN_DOMAIN)

    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)

    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    if os.path.exists(MODEL_PATH):
        agent.load_state_dict(torch.load(MODEL_PATH))
        agent.eval()
        print("✅ Loaded trained PPO policy")
    else:
        print("⚠️ No trained policy found")

    query = "treatment for diabetes"
    state = env.reset(query)

    print("\nInitial Query:", query)

    initial_results = retriever.retrieve(query)
    if initial_results:
        print("Initial Top Result:", initial_results[0][1])

    action_fn, _, _ = agent.select_action(state["vector"])
    next_state, reward, done, info = env.step(state, action_fn)

    print("\nChosen Action:", action_fn.__name__)
    print("Rewritten Query:", info["query"])

    if info["after"]:
        print("Top Result After Rewrite:", info["after"][0][1])

    print("Reward:", round(reward, 3))


if __name__ == "__main__":
    main()
