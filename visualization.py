import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# =========================
# PCA
# =========================

def perform_pca(
    region_points,
    greedy_points
):

    pca = PCA(n_components=2)

    all_points = np.concatenate(
        [
            region_points,
            greedy_points
        ],
        axis=0
    )

    pca.fit(all_points)

    region_2d = pca.transform(
        region_points
    )

    greedy_2d = pca.transform(
        greedy_points
    )

    return region_2d, greedy_2d

# =========================
# Score curve
# =========================

def plot_score_curve(
    r_history,
    g_history
):

    plt.figure(figsize=(7,5))

    plt.plot(
        r_history,
        linewidth=2,
        label='Region'
    )

    plt.plot(
        g_history,
        linewidth=2,
        label='Greedy'
    )

    plt.xlabel("Search step")

    plt.ylabel("Best score")

    plt.title(
        "Performance dynamics"
    )

    plt.legend()

    plt.savefig(
        "figures/score_curve.png",
        dpi=300
    )

    plt.close()

# =========================
# Density map
# =========================

def plot_density(
    region_2d,
    greedy_2d
):

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(12,5)
    )

    axs[0].hist2d(
        region_2d[:,0],
        region_2d[:,1],
        bins=50
    )

    axs[0].set_title(
        "Region search density"
    )

    axs[1].hist2d(
        greedy_2d[:,0],
        greedy_2d[:,1],
        bins=50
    )

    axs[1].set_title(
        "Greedy search density"
    )

    plt.savefig(
        "figures/density.png",
        dpi=300
    )

    plt.close()

# =========================
# Metric curves
# =========================

def plot_metric(
    t_r,
    y_r,
    t_g,
    y_g,
    ylabel,
    filename
):

    plt.figure(figsize=(7,5))

    plt.plot(
        t_r,
        y_r,
        linewidth=2,
        label='Region'
    )

    plt.plot(
        t_g,
        y_g,
        linewidth=2,
        label='Greedy'
    )

    plt.xlabel("Search step")

    plt.ylabel(ylabel)

    plt.title(ylabel)

    plt.legend()

    plt.savefig(
        f"figures/{filename}",
        dpi=300
    )

    plt.close()