import numpy as np

# =========================
# Core parameters
# =========================

n_s = 7
n_o = 2

strength = 3.0

budget_per_run = 11000
samples_per_region = 20
max_steps = 300

# search strategy
region_top_k = 600
greedy_top_k = 1

# search space
dim = n_o * n_s

S_low = -1.5 * np.ones(dim)
S_high = 1.5 * np.ones(dim)

# visualization
trajectory_stride = 30

# automatic interesting case detection
interesting_gap = 0.25