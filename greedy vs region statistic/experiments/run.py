import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import numpy as np

from scipy import stats

from src.search import (
    generate_ground_truth,
    run_search
)

from src.config import *

from src.visualization import plot_results


region_bests = []

greedy_bests = []


for i in range(n_runs):

    seed = np.random.randint(0, 10000)

    np.random.seed(seed)

    current_true_s = generate_ground_truth()

    np.random.seed(seed)

    r_score, _ = run_search(
        budget_per_run,
        current_true_s,
        top_k=600
    )

    region_bests.append(r_score)

    np.random.seed(seed)

    g_score, _ = run_search(
        budget_per_run,
        current_true_s,
        top_k=1
    )

    greedy_bests.append(g_score)

    print(
        f"Run {i+1}: "
        f"Region={r_score:.3f}, "
        f"Greedy={g_score:.3f}"
    )


t_stat, p_val = stats.ttest_rel(
    region_bests,
    greedy_bests
)

print(f"\nPaired T-test p-value: {p_val:.6f}")


plot_results(
    region_bests,
    greedy_bests
)