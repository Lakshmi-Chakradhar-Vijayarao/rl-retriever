def recall_at_k(results, gold_doc, k=3):
    docs = [doc for _, doc in results[:k]]
    return 1.0 if gold_doc in docs else 0.0


def mean_rank(results, gold_doc):
    for i, (_, doc) in enumerate(results):
        if doc == gold_doc:
            return i + 1
    return len(results) + 1
