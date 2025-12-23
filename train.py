from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOQueryRewriteAgent


def load_corpus(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def train():
    corpus = load_corpus("data/corpus.txt")
    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)
    agent = PPOQueryRewriteAgent()

    query = "treatment for diabetes"
    state = env.reset(query)

    rewards = []

    for episode in range(10):
        action = agent.select_action()
        next_state, reward, info = env.step(state, action)

        rewards.append(reward)
        state = next_state

        print(f"Episode {episode}")
        print("Action:", action.__name__)
        print("Reward:", reward)
        print("-" * 30)

    
    agent.update_policy(rewards)

    print("Updated action probabilities:", agent.action_probs)


if __name__ == "__main__":
    train()
