from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOQueryRewriteAgent
from train import load_corpus


def main():
    corpus = load_corpus("data/corpus.txt")
    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)
    agent = PPOQueryRewriteAgent()

    query = "treatment for diabetes"
    state = env.reset(query)

    print("\nInitial Query:", query)
    print("Initial Top Result:", state["results"][0][1])

    action = agent.select_action()
    _, reward, info = env.step(state, action)

    print("\nChosen Action:", action.__name__)
    print("Rewritten Query:", info["rewritten_query"])
    print("Top Result After Rewrite:", info["after_results"][0][1])
    print("Reward:", reward)


if __name__ == "__main__":
    main()
