from reward import compute_reward


class QueryRewriteEnv:
    """
    Reinforcement Learning environment for query rewriting.
    The environment evaluates how a rewritten query affects
    downstream retrieval quality.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def reset(self, query, top_k=3):
        """
        Initialize the environment with an initial query.
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "results": results
        }

    def step(self, state, action_fn, top_k=3):
        """
        Take one environment step.

        Args:
            state: dict containing current query and retrieval results
            action_fn: function that rewrites the query
            top_k: number of documents to retrieve

        Returns:
            next_state: updated state after applying the action
            reward: scalar reward signal
            info: detailed diagnostics for analysis/debugging
        """

        original_query = state["query"]

        # Apply action (query rewrite)
        rewritten_query = action_fn(original_query)

        # Retrieve before and after rewrite
        before_results = self.retriever.retrieve(original_query, top_k=top_k)
        after_results = self.retriever.retrieve(rewritten_query, top_k=top_k)

        # Compute shaped reward (retrieval + semantic preservation)
        reward = compute_reward(
            before_results=before_results,
            after_results=after_results,
            original_query=original_query,
            rewritten_query=rewritten_query
        )

        next_state = {
            "query": rewritten_query,
            "results": after_results
        }

        info = {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "before_results": before_results,
            "after_results": after_results
        }

        return next_state, reward, info
