# Dynamics in Attractor Landscapes

Code accompanying the manuscipt:

"Localized Reward-Conditioned Information Usage Reduces Exploration Diversity and Limits Long-Term Optimization in Fragmented Nonlinear Environments"

## Overview

This repository contains the code,data and figures used in the study of comparation a local greedy search strategy with a distributed regional exploration strategies in fragmented optimization landscapes.

## Main Findings

* Different search strategies produce distinct exploration structures in fragmented search landscapes.

* Greedy search rapidly concentrates sampling around a small number of regions, whereas region strategy maintains distributed exploration across multiple candidate regions.

* PCA trajectory analysis reveals systematic differences in the spatial organization of search trajectories.

* These structural differences are associated with differences in long-term optimization performance, suggesting that maintaining distributed regional exploration can improve information acquisition in fragmented environments.

## Repository Structure

greedy vs region statistic/

* Generate new greedy and region statistical difference in optimization.

greedy vs region trajectory/

* Analyze previously generated data.
* Compute landscape metrics.
* Generate PCA and visualization figures.

results/

* Experimental outputs used for analysis and visualization.

paper/

* Preprint manuscript.

## Requirements

Python 3.11+

Required packages:

* numpy
* scipy
* matplotlib
* scikit-learn
* sys
* os
## Reproducing Figures

To reproduce the visualizations from the stored experimental data:
open greedy vs region trajectory folder
py src\analyze_landscape2.py

## Running New Experiments

To generate new optimization trajectories:
open greedy vs region trajectory folder
py src\main.py
The generated outputs can then be analyzed using the scripts in the greedy vs region trajectory directory.

To generate new statistical difference performance:
open greedy vs region statistic folder
py experiments\run.py

## Notes

Scripts inside the greedy vs region trajectory directory should be executed from the greedy vs region trajectory folder rather than the repository root because relative paths are used for loading data.
