# run_perturbation_magnitude_sweep.py
"""
Sweep global edge-weight reduction magnitudes relative to the healthy matrix:
  p in {0.10, 0.20, 0.30, 0.50}
For each p, simulate across 30 seeds and record the final R. Plot mean±CI.

Outputs:
  - results/figures/finalR_vs_perturbation.png
  - results/tables/finalR_vs_perturbation.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ql_utils import ensure_adjacency_csvs, SEED_LIST, FIG_DIR, TAB_DIR, _normalize_symmetric
from kuramoto import simulate_kuramoto

plt.rcParams.update({
    "figure.figsize": (6.2, 4.2),
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def mean_ci(vals, alpha=0.05):
    m = np.mean(vals)
    s = np.std(vals, ddof=1)
    z = 1.96
    return m, z * s / np.sqrt(len(vals))

def main():
    pair = ensure_adjacency_csvs()
    H = pair.healthy.copy()

    T = 1000
    dt = 0.01
    K = 2.0
    sigma = 0.05

    percents = [0.10, 0.20, 0.30, 0.50]
    means, cis = [], []
    rows = [["percent_reduction", "mean_final_R", "std_final_R", "n_seeds"]]

    for p in percents:
        A_p = _normalize_symmetric((1.0 - p) * H)
        finals = []
        for seed in SEED_LIST:
            out = simulate_kuramoto(A_p, T=T, dt=dt, K=K, sigma=sigma, rng=np.random.default_rng(seed))
            finals.append(out["R"][-1])
        finals = np.array(finals)
        m, ci = mean_ci(finals)
        means.append(m)
        cis.append(ci)
        rows.append([p, float(np.mean(finals)), float(np.std(finals, ddof=1)), len(finals)])

    x = np.array([int(p*100) for p in percents])
    fig, ax = plt.subplots()
    ax.errorbar(x, means, yerr=cis, fmt='o-', color="#9467bd", capsize=3, label="Final R ± 95% CI")
    ax.set_xlabel("Global edge-weight reduction (%)")
    ax.set_ylabel("Final order parameter R")
    ax.set_title("Dose–response of Synchronization to Structural Perturbation")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    out_fig = os.path.join(FIG_DIR, "finalR_vs_perturbation.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"[SAVED] {out_fig}")

    out_tab = os.path.join(TAB_DIR, "finalR_vs_perturbation.csv")
    with open(out_tab, "w") as f:
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")
    print(f"[SAVED] {out_tab}")

if __name__ == "__main__":
    main()