# PPO settings
STATE_DIM = 4
ACTION_DIM = 4
NUM_EPISODES = 30
GAMMA = 0.99

# Environment
MAX_STEPS = 3
TOP_K = 3

# Training / Demo domain
TRAIN_DOMAIN = "medical"

# Reward weights (RLHF-style)
REWARD_WEIGHTS = {
    "rank": 1.0,
    "semantic": 0.5,
    "grounding": 0.3,
    "length_penalty": 0.02
}

# Paths
MODEL_PATH = "models/policy.pt"
RESULTS_PATH = "results/metrics.csv"
