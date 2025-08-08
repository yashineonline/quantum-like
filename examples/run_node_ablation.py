# run_node_ablation.py
"""
Node-wise ablation on top-5 hubs (by degree or eigenvector centrality):
  - Set all incident edges of the node to zero (isolate node)
  - Simulate across seeds and measure change in final R relative to healthy

Outputs:
  - results/figures/node_ablation_deltaR.png
  - results/tables/node_ablation_deltaR.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ql_utils import ensure_adjacency_csvs, top_hubs_by, SEED_LIST, FIG_DIR, TAB_DIR
from kuramoto import simulate_kuramoto

plt.rcParams.update({
    "figure.figsize": (7.2, 4.2),
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def ablate_node(A: np.ndarray, idx: int) -> np.ndarray:
    B = A.copy()
    B[idx, :] = 0.0
    B[:, idx] = 0.0
    return B

def run_final_R(A: np.ndarray, T=1000, dt=0.01, K=2.0, sigma=0.05) -> np.ndarray:
    finals = []
    for seed in SEED_LIST:
        out = simulate_kuramoto(A, T=T, dt=dt, K=K, sigma=sigma, rng=np.random.default_rng(seed))
        finals.append(out["R"][-1])
    return np.array(finals)

def main(mode="degree"):
    pair = ensure_adjacency_csvs()
    H = pair.healthy

    # Baseline healthy finals
    finals_healthy = run_final_R(H)
    base_mean = float(np.mean(finals_healthy))
    print(f"[INFO] Baseline healthy final R mean over seeds: {base_mean:.3f}")

    hubs = top_hubs_by(H, mode=mode, k=5)
    delta_means, delta_cis = [], []
    rows = [["node_index", "delta_mean_final_R", "delta_std_final_R", "n_seeds"]]

    for node in hubs:
        A_ab = ablate_node(H, node)
        finals_ab = run_final_R(A_ab)
        delta = finals_ab - finals_healthy  # change vs baseline per seed
        m = float(np.mean(delta))
        s = float(np.std(delta, ddof=1))
        n = len(delta)
        ci = 1.96 * s / np.sqrt(n)
        delta_means.append(m)
        delta_cis.append(ci)
        rows.append([int(node), m, s, n])
        print(f"[ABLATE] node={node:3d}  ΔR_mean={m:.3f}  (±{ci:.3f})")

    # Plot
    x = np.arange(len(hubs))
    fig, ax = plt.subplots()
    ax.bar(x, delta_means, yerr=delta_cis, color="#8c564b", alpha=0.9, capsize=3)
    ax.set_xticks(x, [str(i) for i in hubs])
    ax.set_xlabel("Node (hub index)")
    ax.set_ylabel("Δ Final R (ablation - baseline)")
    ax.set_title(f"Impact of Ablating Top-5 Hubs by {mode.title()}")
    out_fig = os.path.join(FIG_DIR, "node_ablation_deltaR.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"[SAVED] {out_fig}")

    out_tab = os.path.join(TAB_DIR, "node_ablation_deltaR.csv")
    with open(out_tab, "w") as f:
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")
    print(f"[SAVED] {out_tab}")

if __name__ == "__main__":
    main(mode="degree")