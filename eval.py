def recall_at_k(results, relevant_doc, k=3):
    docs = [doc for _, doc in results[:k]]
    return 1.0 if relevant_doc in docs else 0.0


def mean_rank(results, relevant_doc):
    for i, (_, doc) in enumerate(results):
        if relevant_doc == doc:
            return i + 1
    return len(results) + 1
