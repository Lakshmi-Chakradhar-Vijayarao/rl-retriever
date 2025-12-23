from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=3):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self.corpus),
            key=lambda x: x[0],
            reverse=True
        )
        return ranked[:top_k]
