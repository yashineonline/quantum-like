# run_adjacency_diagnostics.py
"""
Quick diagnostics and visuals for the input adjacencies:
  - Degree distributions
  - Matrix heatmaps and difference

Outputs:
  - results/figures/adjacency_heatmap_healthy.png
  - results/figures/adjacency_heatmap_perturbed.png
  - results/figures/adjacency_heatmap_difference.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ql_utils import ensure_adjacency_csvs, FIG_DIR

plt.rcParams.update({
    "figure.figsize": (6.0, 5.0),
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def plot_heatmap(A, title, path):
    fig, ax = plt.subplots()
    im = ax.imshow(A, cmap="magma", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weight")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"[SAVED] {path}")

def main():
    pair = ensure_adjacency_csvs()
    plot_heatmap(pair.healthy, "Adjacency (healthy)", os.path.join(FIG_DIR, "adjacency_heatmap_healthy.png"))
    plot_heatmap(pair.perturbed, "Adjacency (perturbed)", os.path.join(FIG_DIR, "adjacency_heatmap_perturbed.png"))
    diff = pair.healthy - pair.perturbed
    plot_heatmap(diff, "Adjacency difference (healthy - perturbed)", os.path.join(FIG_DIR, "adjacency_heatmap_difference.png"))

if __name__ == "__main__":
    main()