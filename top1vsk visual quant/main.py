import numpy as np

from config import *

from search_algorithms import *
from visualization import *
from metrics import *

# =========================
# Find interesting case
# =========================

found = False

trial = 0

while not found:

    trial += 1

    seed = np.random.randint(
        0,
        100000
    )

    np.random.seed(seed)

    true_s = generate_ground_truth()

    # =====================
    # Region search
    # =====================

    np.random.seed(seed)

    (
        r_score,
        r_history,
        r_all_points,
        r_all_scores
    ) = run_search(
        budget_per_run,
        true_s,
        region_top_k
    )

    # =====================
    # Greedy search
    # =====================

    np.random.seed(seed)

    (
        g_score,
        g_history,
        g_all_points,
        g_all_scores
    ) = run_search(
        budget_per_run,
        true_s,
        greedy_top_k
    )

    gap = r_score - g_score

    print(
        f"Trial {trial} | "
        f"Region={r_score:.3f} "
        f"Greedy={g_score:.3f} "
        f"Gap={gap:.3f}"
    )

    if gap > interesting_gap:

        found = True

        print("\nInteresting case found.")
        print("Seed =", seed)

# =========================
# PCA
# =========================

region_2d, greedy_2d = perform_pca(
    r_all_points,
    g_all_points
)

# =========================
# Score dynamics
# =========================

plot_score_curve(
    r_history,
    g_history
)

# =========================
# Density map
# =========================

plot_density(
    region_2d,
    greedy_2d
)

# =========================
# Metrics
# =========================

(
    t_r,
    spread_r,
    entropy_r,
    area_r
) = metrics_over_time(
    region_2d,
    step=100
)

(
    t_g,
    spread_g,
    entropy_g,
    area_g
) = metrics_over_time(
    greedy_2d,
    step=100
)

# =========================
# Metric plots
# =========================

plot_metric(
    t_r,
    spread_r,
    t_g,
    spread_g,
    "Exploration Spread",
    "spread.png"
)

plot_metric(
    t_r,
    entropy_r,
    t_g,
    entropy_g,
    "Search Entropy",
    "entropy.png"
)

plot_metric(
    t_r,
    area_r,
    t_g,
    area_g,
    "Explored Area",
    "area.png"
)

# =========================
# Final statistics
# =========================

print("\n========== FINAL ==========")

print("\nFinal score")
print("Region:", r_score)
print("Greedy:", g_score)

print("\nFinal spread")
print("Region:", spread_r[-1])
print("Greedy:", spread_g[-1])

print("\nFinal entropy")
print("Region:", entropy_r[-1])
print("Greedy:", entropy_g[-1])

print("\nFinal area")
print("Region:", area_r[-1])
print("Greedy:", area_g[-1])

# =========================
# Save
# =========================

np.save(
    "results/r_points.npy",
    r_all_points
)

np.save(
    "results/g_points.npy",
    g_all_points
)
np.save(
    "results/r_scores.npy",
    r_all_scores
)

np.save(
    "results/g_scores.npy",
    g_all_scores
)
print("\nFinished.")