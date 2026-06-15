# Regional Exploration

Code accompanying the preprint:

"Localized Reward-Conditioned Information Usage Reduces Exploration Diversity and Limits Long-Term Optimization in Fragmented Nonlinear Environments"

## Overview

This repository contains:

- greedy vs region score statistical difference
- PCA-based region score heatmap
- score and metrics for measuring exploration curve of two startegy

## Repository Structure

es/            Evolution Strategy baseline

top1vsk/       Regional exploration algorithm

pca/           PCA trajectory analysis

figures/       Generated figures

## Usage

Run ES:

python es/run.py

Run Top1-vs-K:

python top1vsk/run.py

Generate PCA plots:

python pca/pca_analysis.py

