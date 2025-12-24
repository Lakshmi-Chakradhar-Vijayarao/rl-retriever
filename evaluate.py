import json
import torch
import csv
import os
from collections import defaultdict

from retriever import Retriever
from env import QueryRewriteEnv
from agent import PPOAgent
from eval_metrics import recall_at_k, mean_rank
from baselines import no_rewrite, random_rewrite, static_rewrite
from config import STATE_DIM, ACTION_DIM, MODEL_PATH, RESULTS_PATH


def load_queries(path):
    with open(path) as f:
        return json.load(f)


def evaluate_policy(policy_fn, retriever, queries):
    recall_scores = []
    ranks = []

    for q, gold in queries.items():
        rewritten = policy_fn(q)
        results = retriever.retrieve(rewritten)

        recall_scores.append(recall_at_k(results, gold))
        ranks.append(mean_rank(results, gold))

    return (
        sum(recall_scores) / len(recall_scores),
        sum(ranks) / len(ranks)
    )


def main():
    domains = ["medical", "legal", "finance"]
    results = []
    recall_deltas = []

    for domain in domains:
        print(f"\n🔍 Evaluating domain: {domain}")

        corpus = open(f"data/{domain}/corpus.txt").read().splitlines()
        queries = load_queries(f"data/{domain}/queries.json")

        retriever = Retriever(corpus)
        env = QueryRewriteEnv(retriever)

        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        agent.load_state_dict(torch.load(MODEL_PATH))
        agent.eval()

        action_counts = defaultdict(int)

        def ppo_policy(q):
            state = env.reset(q)
            action_fn, _, _ = agent.select_action(state["vector"])
            action_counts[action_fn.__name__] += 1
            return action_fn(q)

        no_rewrite_metrics = evaluate_policy(no_rewrite, retriever, queries)
        random_metrics = evaluate_policy(random_rewrite, retriever, queries)
        static_metrics = evaluate_policy(static_rewrite, retriever, queries)
        ppo_metrics = evaluate_policy(ppo_policy, retriever, queries)

        baseline_recall = no_rewrite_metrics[0]
        ppo_recall = ppo_metrics[0]

        if baseline_recall > 0:
            delta_pct = ((ppo_recall - baseline_recall) / baseline_recall) * 100
            recall_deltas.append(delta_pct)
        else:
            delta_pct = 0.0

        print("PPO action distribution:", dict(action_counts))
        print(f"Recall improvement vs baseline: {delta_pct:.2f}%")

        results.append({
            "domain": domain,
            "no_rewrite": no_rewrite_metrics,
            "random": random_metrics,
            "static": static_metrics,
            "ppo": ppo_metrics,
        })

    if recall_deltas:
        avg_delta = sum(recall_deltas) / len(recall_deltas)
        print(f"\n📈 Average recall improvement (non-saturated domains): {avg_delta:.2f}%")

    os.makedirs("results", exist_ok=True)

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Method", "Recall@3", "MeanRank"])

        for r in results:
            for method, vals in r.items():
                if method == "domain":
                    continue
                writer.writerow([r["domain"], method, vals[0], vals[1]])

    print("\n✅ Evaluation complete. Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
