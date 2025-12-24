def token_overlap(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb) / max(len(sa), 1)


def compute_reward(before, after, original_q, rewritten_q):
    """
    RLHF-style shaped reward:
    - rank improvement
    - semantic preservation
    - grounding quality
    - rewrite penalty
    """

    if not before or not after:
        return 0.0

    # Retrieval improvement (primary task reward)
    rank_gain = after[0][0] - before[0][0]

    # Semantic preservation (dense stabilizer)
    semantic_sim = token_overlap(original_q, rewritten_q)

    # Grounding quality (query ↔ document alignment)
    grounding = token_overlap(rewritten_q, after[0][1])

    # Rewrite penalty (avoid bloated queries)
    length_penalty = max(0, len(rewritten_q.split()) - len(original_q.split())) * 0.02

    reward = (
        1.0 * rank_gain +
        0.5 * semantic_sim +
        0.3 * grounding -
        length_penalty
    )

    return reward
