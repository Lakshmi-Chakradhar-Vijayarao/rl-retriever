# RL-Retriever: Reinforcement Learning–Based Query Optimization for RAG

RL-Retriever is a **research-grade, executable system** that formulates **query rewriting for Retrieval-Augmented Generation (RAG)** as a **reinforcement learning (RL) control problem**.

Instead of treating query formulation as a static or heuristic preprocessing step, this project models query rewriting as a **sequential decision-making task**, where an RL agent learns **when and how to rewrite queries** to improve downstream retrieval quality — and when **not** to rewrite.

The emphasis of this repository is on **correct RL formulation, reward design, and system-level reasoning**, rather than large-scale training or heavyweight frameworks.

---

## Motivation

In modern RAG systems, retrieval quality is often the **primary bottleneck**.
Even strong embedding models can fail when user queries are:

* underspecified
* informal
* ambiguous
* poorly aligned with the corpus vocabulary

Common limitations of existing approaches include:

* static query templates
* heuristic query expansion
* optimization based purely on text similarity
* lack of learning signals tied to retrieval outcomes
* no mechanism to decide *when rewriting is unnecessary*

**Key idea:**

> Query rewriting should be optimized using the same principles as other control and optimization problems — by defining states, actions, rewards, and policies.

RL-Retriever demonstrates this idea in a **clean, interpretable, and reproducible way**.

---

## Problem Formulation (RL Perspective)

The task is formulated as a **Markov Decision Process (MDP)**.

### State

The environment state captures retrieval context:

* current query
* statistics from top-K retrieval results (BM25 scores)
* query length
* step index (for multi-step refinement)

### Action

A discrete, interpretable set of query rewrite actions:

* add domain-specific terms
* expand acronyms / abbreviations
* add guideline or contextual terms
* no-op (explicitly choose not to rewrite)

### Environment

* A BM25-based retriever operating over a document corpus
* Evaluated across **medical, legal, and financial domains**

### Reward (RLHF-style)

A **shaped reward** combining:

* **retrieval rank improvement** (primary signal)
* **semantic preservation** (embedding similarity)
* **grounding quality** (query–document overlap)
* **rewrite penalty** (discourages unnecessary verbosity)

This mirrors **RLHF-style reward shaping**, while keeping the optimization objective grounded in retrieval performance.

### Objective

Learn a policy that:

* improves retrieval for underspecified queries
* avoids harmful rewrites when retrieval is already optimal
* remains stable and interpretable

---

## High-Level Architecture

```
Query
  ↓
State Construction
  ↓
PPO Policy (Action Selection)
  ↓
Query Rewrite
  ↓
Retriever (BM25)
  ↓
Reward Computation
  ↓
Policy Update
```

The system is intentionally modular so each component can be inspected, modified, or replaced independently.

---

## Repository Structure

```
rl-retriever/
├── actions.py          # Discrete query rewrite actions
├── agent.py            # PPO agent (policy + value networks)
├── env.py              # RL environment (state, step, reward)
├── reward.py           # RLHF-style reward computation
├── retriever.py        # BM25-based retriever
├── train.py            # PPO training loop
├── main.py             # Single-query inference demo
├── evaluate.py         # Multi-domain evaluation script
├── eval_metrics.py     # Recall@K and Mean Rank
├── baselines.py        # No-rewrite, static, random baselines
├── config.py           # Centralized configuration
├── data/
│   ├── medical/
│   ├── legal/
│   └── finance/
├── requirements.txt
└── README.md
```

---

## Key Design Decisions (and Why)

### 1. Discrete, Interpretable Actions

Query rewrites are modeled as **explicit actions**, not free-form text generation.

This ensures:

* transparent policy behavior
* explainable learning dynamics
* easy debugging and analysis
* safe deployment behavior (`no_op` is a first-class action)

---

### 2. RLHF-Style Reward Shaping

Pure retrieval rewards are sparse and unstable.
To improve learning stability, the reward combines:

* rank improvement (sparse, task-aligned)
* embedding similarity (dense, stabilizing)
* grounding quality (retrieval relevance)
* rewrite penalties (regularization)

This balances **learning signal strength** with **system correctness**.

---

### 3. PPO for Stability

The agent uses **Proximal Policy Optimization (PPO)** with:

* shared policy/value networks
* advantage-based updates
* conservative policy updates

PPO was chosen for its stability and suitability for small, interpretable environments.

---

### 4. Conservative Policy Learning

The agent explicitly learns when **not** to rewrite.

In domains where baseline retrieval is already optimal, the learned policy prefers `no_op`, preventing regression — a critical requirement for real-world RAG systems.

---

## Example Workflow

### Training

```bash
python train.py
```

Typical output:

```
Episode 12 | Total reward: 20.96
Episode 18 | Total reward: 30.21
Episode 29 | Total reward: 11.53
```

Interpretation:

* rewards increase without collapse
* policy explores but stabilizes
* rewriting is selective, not forced

---

### Inference Demo

```bash
python main.py
```

Example:

```
Initial Query: dm therapy
Chosen Action: add_domain_terms
Rewritten Query: dm therapy diabetes type 2 treatment metformin
Top Result: Metformin is the first line treatment for type 2 diabetes.
Reward: 2.39
```

---

## Evaluation

Evaluation is performed across **medical, legal, and financial domains**, comparing:

* no rewrite
* random rewrite
* static rewrite
* PPO-based rewrite

Metrics:

* Recall@3
* Mean Rank

### Key Observations

* PPO improves recall on **underspecified queries**
* PPO avoids rewriting when retrieval is already optimal
* No regression on saturated domains
* Action distributions are interpretable and auditable

---

## Results Summary

* Achieved **up to +33% Recall@3 improvement** on medical queries
* Demonstrated safe fallback (`no_op`) on legal and financial domains
* Validated RL-based query rewriting as a controllable system component

---

## Why This Matters

This project demonstrates that:

* RL can be applied **before generation**, not only during generation
* Retrieval can be treated as a **first-class optimization objective**
* System-level design often matters more than model scale
* Conservative RL policies are essential for production RAG systems

The same formulation generalizes to:

* domain-specific retrieval optimization
* adaptive information access
* control problems in distributed systems

---

## Limitations & Future Work

* Small, illustrative corpora
* BM25-only retrieval backend
* Single-step rewrite per episode

Possible extensions:

* multi-step query refinement
* hybrid BM25 + embedding retrieval
* DQN-based discrete policies
* online learning with user feedback

---

## Notes

* Training artifacts are excluded by design
* Emphasis is on **clarity, correctness, and reproducibility**
* This is a research prototype, not a production system

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer | Reinforcement Learning | RAG | Systems & Optimization

---
