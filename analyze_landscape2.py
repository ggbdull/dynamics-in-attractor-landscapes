import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d

# =========================
# Load
# =========================

r_points = np.load(
    "results/r_points.npy"
)

g_points = np.load(
    "results/g_points.npy"
)

r_scores = np.load(
    "results/r_scores.npy"
)

g_scores = np.load(
    "results/g_scores.npy"
)

# =========================
# PCA
# =========================

pca = PCA(n_components=2)

all_points = np.concatenate(
    [r_points, g_points],
    axis=0
)

pca.fit(all_points)

r_2d = pca.transform(r_points)

g_2d = pca.transform(g_points)
# PCA explained variance
var_ratio = pca.explained_variance_ratio_

pc1_var = var_ratio[0] * 100
pc2_var = var_ratio[1] * 100
# =========================
# Shared axis range
# =========================

xmin = min(
    r_2d[:,0].min(),
    g_2d[:,0].min()
)

xmax = max(
    r_2d[:,0].max(),
    g_2d[:,0].max()
)

ymin = min(
    r_2d[:,1].min(),
    g_2d[:,1].min()
)

ymax = max(
    r_2d[:,1].max(),
    g_2d[:,1].max()
)

# =========================
# Bin landscape
# =========================

bins = 70

# Region
r_stat, xedge, yedge, _ = binned_statistic_2d(
    r_2d[:,0],
    r_2d[:,1],
    r_scores,
    statistic='mean',
    bins=bins,
    range=[
        [xmin, xmax],
        [ymin, ymax]
    ]
)

# Greedy
g_stat, _, _, _ = binned_statistic_2d(
    g_2d[:,0],
    g_2d[:,1],
    g_scores,
    statistic='mean',
    bins=bins,
    range=[
        [xmin, xmax],
        [ymin, ymax]
    ]
)

# =========================
# Mask unexplored regions
# =========================

r_masked = np.ma.masked_invalid(
    r_stat.T
)

g_masked = np.ma.masked_invalid(
    g_stat.T
)

# =========================
# Colormap
# =========================

cmap = plt.cm.viridis.copy()

# unexplored region = white
cmap.set_bad(color='white')

# =========================
# Plot
# =========================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(14,6),
    sharex=True,
    sharey=True
)

# =========================
# Region
# =========================

im1 = axs[0].imshow(
    r_masked,
    origin='lower',
    extent=[
        xmin,
        xmax,
        ymin,
        ymax
    ],
    aspect='auto',
    cmap=cmap,
    vmin=-0.5,
    vmax=1.0
)

# 叠加采样点
axs[0].scatter(
    r_2d[:,0],
    r_2d[:,1],
    c=r_scores,
    cmap='viridis',
    s=8,
    alpha=0.35,
    edgecolors='none',
    vmin=-0.5,
    vmax=1.0
)

axs[0].set_title(
    "Region search"
)

# =========================
# Greedy
# =========================

im2 = axs[1].imshow(
    g_masked,
    origin='lower',
    extent=[
        xmin,
        xmax,
        ymin,
        ymax
    ],
    aspect='auto',
    cmap=cmap,
    vmin=-0.5,
    vmax=1.0
)

axs[1].scatter(
    g_2d[:,0],
    g_2d[:,1],
    c=g_scores,
    cmap='viridis',
    s=8,
    alpha=0.35,
    edgecolors='none',
    vmin=-0.5,
    vmax=1.0
)

axs[1].set_title(
    "Greedy search"
)

# =========================
# Labels
# =========================

for ax in axs:

    ax.set_xlabel(
    f"PC1 ({pc1_var:.1f}% variance)"
)

ax.set_ylabel(
    f"PC2 ({pc2_var:.1f}% variance)"
)

# =========================
# Colorbar
# =========================

cax = fig.add_axes(
    [0.92, 0.2, 0.02, 0.6]
)

cbar = fig.colorbar(
    im2,
    cax=cax
)

cbar.set_label(
    "Mean local score"
)

# =========================
# Layout
# =========================

plt.subplots_adjust(
    right=0.9,
    wspace=0.15
)

plt.savefig(
    "figures/landscape_mean_score.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()