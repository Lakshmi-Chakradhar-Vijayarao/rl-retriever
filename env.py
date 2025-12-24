from reward import compute_reward


class QueryRewriteEnv:
    """
    PPO-compatible RL environment for query rewriting.
    """

    def __init__(self, retriever, max_steps=3):
        self.retriever = retriever
        self.max_steps = max_steps
        self.step_t = 0
        self.original_query = None

    def reset(self, query, top_k=3):
        self.step_t = 0
        self.original_query = query
        results = self.retriever.retrieve(query, top_k)

        return self._build_state(query, results)

    def step(self, state, action_fn, top_k=3):
        self.step_t += 1

        current_query = state["query"]

        rewritten_query = action_fn(current_query)

        before = self.retriever.retrieve(current_query, top_k)
        after = self.retriever.retrieve(rewritten_query, top_k)

        reward = compute_reward(
            before,
            after,
            self.original_query,
            rewritten_query
        )

        done = self.step_t >= self.max_steps

        next_state = self._build_state(rewritten_query, after)

        info = {
            "before": before,
            "after": after,
            "query": rewritten_query
        }

        return next_state, reward, done, info

    def _build_state(self, query, results):
        top_score = results[0][0] if results else 0.0
        avg_score = sum(r[0] for r in results) / max(len(results), 1)

        return {
            "query": query,
            "vector": [
                top_score,
                avg_score,
                len(query.split()),
                self.step_t
            ]
        }
