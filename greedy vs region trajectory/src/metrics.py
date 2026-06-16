import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

# =========================
# Spread
# =========================

def compute_spread(points):

    if len(points) < 2:
        return 0

    return np.mean(
        pdist(points)
    )

# =========================
# Entropy
# =========================

def compute_entropy(
    points,
    bins=30
):

    H, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins
    )

    P = H / np.sum(H)

    entropy = -np.sum(
        P * np.log(P + 1e-12)
    )

    return entropy

# =========================
# Explored area
# =========================

def compute_area(points):

    if len(points) < 3:
        return 0

    try:

        hull = ConvexHull(points)

        return hull.volume

    except:

        return 0

# =========================
# Dynamics over time
# =========================

def metrics_over_time(
    points_2d,
    step=100
):

    timesteps = []

    spreads = []

    entropies = []

    areas = []

    for t in range(
        step,
        len(points_2d),
        step
    ):

        current = points_2d[:t]

        timesteps.append(t)

        spreads.append(
            compute_spread(current)
        )

        entropies.append(
            compute_entropy(current)
        )

        areas.append(
            compute_area(current)
        )

    return (
        timesteps,
        spreads,
        entropies,
        areas
    )