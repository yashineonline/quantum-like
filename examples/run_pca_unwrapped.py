# run_pca_unwrapped.py
"""
PCA on unwrapped phase histories for healthy and perturbed graphs.
- Runs one representative seed (configurable)
- Produces scree plots and PC1 spatial loadings

Outputs:
  - results/figures/pca_scree_healthy.png
  - results/figures/pca_scree_perturbed.png
  - results/figures/pca_pc1_loadings_healthy.png
  - results/figures/pca_pc1_loadings_perturbed.png
  - results/tables/pca_explained_variance_{condition}.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ql_utils import ensure_adjacency_csvs, FIG_DIR, TAB_DIR
from kuramoto import simulate_kuramoto, unwrap_time_series

plt.rcParams.update({
    "figure.figsize": (7.5, 4.0),
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def pca_svd(X: np.ndarray):
    """
    Column-centered PCA via SVD.
    X: [T, N]. Returns (singular_values, V) where columns of V are loadings.
    """
    Xc = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return S, Vt.T  # V

def run_and_pca(A: np.ndarray, seed: int, label: str):
    out = simulate_kuramoto(A, T=1000, dt=0.01, K=2.0, sigma=0.05, rng=np.random.default_rng(seed))
    unwrap = unwrap_time_series(out["theta"])
    S, V = pca_svd(unwrap)
    var = S**2
    ratio = var / np.sum(var)

    # Scree
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(ratio)+1), ratio, 'o-', color="#2ca02c")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PCA Scree ({label})")
    path1 = os.path.join(FIG_DIR, f"pca_scree_{label}.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=300)
    print(f"[SAVED] {path1}")

    # PC1 loadings
    pc1 = V[:, 0]
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(pc1)), pc1, color="#1f77b4")
    ax.set_xlabel("Node index")
    ax.set_ylabel("PC1 loading")
    ax.set_title(f"PC1 Spatial Loadings ({label})")
    path2 = os.path.join(FIG_DIR, f"pca_pc1_loadings_{label}.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=300)
    print(f"[SAVED] {path2}")

    # Table
    out_tab = os.path.join(TAB_DIR, f"pca_explained_variance_{label}.csv")
    with open(out_tab, "w") as f:
        f.write("component,explained_variance_ratio\n")
        for i, r in enumerate(ratio, start=1):
            f.write(f"{i},{r}\n")
    print(f"[SAVED] {out_tab}")

def main():
    pair = ensure_adjacency_csvs()
    seed = 123  # representative; change as needed for replicates
    run_and_pca(pair.healthy, seed, "healthy")
    run_and_pca(pair.perturbed, seed, "perturbed")

if __name__ == "__main__":
    main()