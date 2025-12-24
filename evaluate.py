from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOAgent
from baselines import no_rewrite, random_rewrite, static_rewrite
from eval import recall_at_k, mean_rank
from train import load_corpus


QUERIES = {
    "treatment for diabetes": "Metformin is the first line treatment for type 2 diabetes."
}


def evaluate_policy(policy_fn, env, retriever):
    recall_scores = []
    ranks = []

    for query, gold_doc in QUERIES.items():
        baseline_results = retriever.retrieve(query)
        rewritten_query = policy_fn(query)
        results = retriever.retrieve(rewritten_query)

        recall_scores.append(recall_at_k(results, gold_doc))
        ranks.append(mean_rank(results, gold_doc))

    return sum(recall_scores)/len(recall_scores), sum(ranks)/len(ranks)


def main():
    corpus = load_corpus("data/corpus.txt")
    retriever = Retriever(corpus)
    env = QueryRewriteEnv(retriever)

    agent = PPOAgent(state_dim=4, action_dim=4)
    agent.load_state_dict(__import__("torch").load("models/policy.pt"))
    agent.eval()

    def ppo_policy(q):
        state = env.reset(q)
        action_fn, _, _ = agent.select_action(state["vector"])
        return action_fn(q)

    print("\nEvaluation Results")
    print("------------------")
    print("No Rewrite:", evaluate_policy(no_rewrite, env, retriever))
    print("Random Rewrite:", evaluate_policy(random_rewrite, env, retriever))
    print("Static Rewrite:", evaluate_policy(static_rewrite, env, retriever))
    print("PPO Rewrite:", evaluate_policy(ppo_policy, env, retriever))


if __name__ == "__main__":
    main()
