def token_overlap(q1, q2):
    s1 = set(q1.lower().split())
    s2 = set(q2.lower().split())
    return len(s1 & s2) / max(len(s1), 1)


def compute_reward(before_results, after_results, original_query, rewritten_query):
    """
    Reward = retrieval gain + semantic preservation
    """

    if not before_results or not after_results:
        return 0.0

    # Retrieval improvement (primary signal)
    retrieval_gain = after_results[0][0] - before_results[0][0]

    # Semantic preservation (dense signal)
    semantic_bonus = token_overlap(original_query, rewritten_query)

    # Weighted reward
    reward = 0.7 * retrieval_gain + 0.3 * semantic_bonus

    return reward
