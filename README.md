# RL-Retriever: Reinforcement Learning–Based Query Optimization for RAG

RL-Retriever is a **lightweight, executable research project** that formulates **query rewriting for retrieval-augmented generation (RAG)** as a **reinforcement learning (RL) problem**.

Instead of treating query formulation as a static or heuristic process, this project models it as a **sequential decision-making task**, where an agent learns to rewrite queries in order to **maximize downstream retrieval quality**.

The emphasis of this repository is on **correct RL formulation, reward design, and system-level reasoning**, rather than large-scale training or heavy frameworks.

---

## Motivation

In RAG systems, retrieval quality is often the **primary bottleneck**.
Even strong embedding models fail when queries are underspecified, ambiguous, or poorly aligned with the corpus.

Common limitations of existing approaches:

* static query templates
* heuristic query expansion
* optimization focused on text similarity rather than downstream utility
* lack of learning signals tied to retrieval performance

**Key idea:**

> Query rewriting should be optimized using the same principles as other control and optimization problems — by defining states, actions, rewards, and policies.

RL-Retriever demonstrates this idea in a **clean, minimal, and reproducible way**.

---

## Problem Formulation (RL Perspective)

The task is formulated as a **Markov Decision Process (MDP)**:

### State

* Current query
* Top-K retrieval results from the corpus

### Action

* Apply a query rewrite function
  (e.g., add domain terms, expand acronyms, add guideline context, or do nothing)

### Environment

* A BM25-based retriever operating over a document corpus

### Reward

A **shaped reward** combining:

* **retrieval improvement** (primary signal)
* **semantic preservation** between original and rewritten queries (dense signal)

### Objective

Learn a policy that increases the likelihood of actions that **improve downstream retrieval quality**, while maintaining stability and avoiding degenerate rewrites.

---

## High-Level Architecture

```
Query
  ↓
Action Selection (Query Rewrite)
  ↓
Retriever (BM25)
  ↓
Reward Computation
  ↓
Policy Update
```

The system is intentionally modular, making each component easy to inspect, reason about, and extend.

---

## Repository Structure

```
rl-retriever/
├── main.py          # Executable demo (single-run inference)
├── train.py         # Lightweight RL training loop
├── env.py           # RL environment (state, transition, reward)
├── agent.py         # PPO-style policy agent
├── actions.py       # Discrete query rewrite actions
├── retriever.py     # BM25-based retriever
├── reward.py        # Shaped reward definition
├── data/
│   ├── corpus.txt   # Example document corpus
│   └── queries.txt  # Example input queries
└── requirements.txt
```

---

## Key Design Decisions (and Why)

### 1. Discrete Action Space

Query rewrites are modeled as **explicit, interpretable actions**, not opaque text generation.

This makes:

* policy behavior transparent
* learning dynamics explainable
* debugging straightforward

---

### 2. Shaped Reward (Sparse → Dense)

Pure retrieval-based rewards are often sparse.
To improve learning stability, the reward combines:

* **Retrieval gain** (difference in top-ranked BM25 score)
* **Semantic preservation** (token overlap between original and rewritten query)

This mirrors **RLHF-style reward shaping** while keeping the optimization objective grounded in system performance.

---

### 3. Lightweight PPO-Style Policy

The policy update is intentionally simple:

* action probabilities are reinforced when rewards are positive
* probabilities are normalized to preserve exploration
* no deep networks or heavy frameworks are used

This keeps the focus on **formulation correctness**, not training scale.

---

### 4. Executability Over Scale

The project is designed to:

* run end-to-end with a single command
* be deterministic and interpretable
* demonstrate ideas clearly in a live setting

This makes it ideal for:

* research discussions
* PhD interviews
* system design evaluations

---

## Example Workflow

### Training

```bash
python train.py
```

Example output:

```
Episode 2
Action: add_domain_terms
Reward: 2.39
------------------------------
Episode 5
Action: expand_acronyms
Reward: 0.30
------------------------------
Updated action probabilities: [0.27, 0.36, 0.20, 0.17]
```

Interpretation:

* only certain rewrites materially improve retrieval
* dense rewards stabilize learning
* policy shifts toward high-impact actions without collapsing exploration

---

### Demo (Inference)

```bash
python main.py
```

Example output:

```
Initial Query: treatment for diabetes
Chosen Action: add_domain_terms
Rewritten Query: treatment for diabetes type 2 metformin first line treatment
Top Result After Rewrite: Metformin is the first line treatment for type 2 diabetes.
Reward: 2.39
```

This demonstrates **end-to-end optimization** of retrieval via learned query rewriting.

---

## Why This Matters

This project shows that:

* RL can be applied meaningfully **before generation**, not just during generation
* retrieval quality can be treated as a **first-class optimization objective**
* system-level design choices matter more than model size

The same formulation generalizes to:

* retrieval optimization in other domains (legal, finance)
* adaptive information access
* control and optimization problems in distributed systems

---

## Limitations & Future Work

* Policy is intentionally simple (no neural networks)
* Corpus is small and illustrative
* Future extensions could include:

  * learned policies with PPO/DQN
  * embedding-based retrieval signals
  * multi-step query refinement
  * application to non-text control problems

---

## Notes

* Virtual environments and cached files are excluded by design
* The repository emphasizes **clarity, correctness, and reproducibility**
* This is a research demo, not a production system

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer | Reinforcement Learning | RAG | Systems & Optimization

---

