from config import REWARD_WEIGHTS

# ----------------------------
# Training-safe semantic proxy
# ----------------------------
def token_overlap(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb) / max(len(sa), 1)


def compute_reward(
    before,
    after,
    original_q,
    rewritten_q,
    use_embedding=False
):
    """
    RLHF-style reward:
    - Rank improvement
    - Semantic similarity (token or embedding)
    - Grounding quality
    - Rewrite penalty
    """

    if not before or not after:
        return 0.0

    # 1. Rank improvement
    rank_gain = after[0][0] - before[0][0]

    # 2. Semantic similarity
    if use_embedding:
        semantic_sim = embedding_similarity(original_q, rewritten_q)
    else:
        semantic_sim = token_overlap(original_q, rewritten_q)

    # 3. Grounding quality
    grounding = token_overlap(rewritten_q, after[0][1])

    # 4. Length penalty
    penalty = max(
        0,
        len(rewritten_q.split()) - len(original_q.split())
    ) * REWARD_WEIGHTS["length_penalty"]

    reward = (
        REWARD_WEIGHTS["rank"] * rank_gain
        + REWARD_WEIGHTS["semantic"] * semantic_sim
        + REWARD_WEIGHTS["grounding"] * grounding
        - penalty
    )

    return reward


# ----------------------------
# Embedding similarity (EVAL ONLY)
# ----------------------------
def embedding_similarity(q1, q2):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")

    e1 = model.encode(q1)
    e2 = model.encode(q2)

    return float(
        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    )
