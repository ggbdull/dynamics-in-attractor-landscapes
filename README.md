# Search Dynamics in Attractor Landscapes

This project studies search behavior in high-dimensional non-convex systems with multiple attractor basins.

We compare:

- Local gradient-like search
- Random exploration
- Region-based distributed search

The goal is to understand how different search strategies behave under deceptive attractor landscapes and sparse reward structures.

---

## Main Hypothesis

Local search methods may become trapped in local attractor basins, while region-based distributed exploration can better escape deceptive structures and discover globally favorable states.

---

## Repository Structure

src/             core algorithms  
experiments/     runnable experiments  
figures/         generated figures  
results/         saved outputs  

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt