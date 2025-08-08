# run_param_sweep_heatmaps.py
"""
Parameter sweep across coupling K and noise sigma.
For each (K, sigma), compute:
  - steady-state R: average of R(t) over the last window
  - largest Lyapunov exponent (approximate)

Outputs:
  - results/figures/heatmap_Rss_healthy.png
  - results/figures/heatmap_Rss_perturbed.png
  - results/figures/heatmap_LLE_healthy.png
  - results/figures/heatmap_LLE_perturbed.png
  - results/tables/grid_Rss_{condition}.csv
  - results/tables/grid_LLE_{condition}.csv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ql_utils import ensure_adjacency_csvs, FIG_DIR, TAB_DIR
from kuramoto import simulate_kuramoto, estimate_lyapunov

plt.rcParams.update({
    "figure.figsize": (6.4, 4.8),
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def compute_grid(A, K_vals, S_vals, T=1500, dt=0.01, window=300, seeds=(123, 131, 137)):
    Rss = np.zeros((len(S_vals), len(K_vals)))
    LLE = np.zeros_like(Rss)
    for i, sigma in enumerate(S_vals):
        for j, K in enumerate(K_vals):
            # Average steady-state R across seeds
            finals = []
            for seed in seeds:
                out = simulate_kuramoto(A, T=T, dt=dt, K=K, sigma=sigma, rng=np.random.default_rng(seed))
                finals.append(np.mean(out["R"][-window:]))
            Rss[i, j] = float(np.mean(finals))

            # LLE with lower T for efficiency (noise shared)
            lle = estimate_lyapunov(A, T=3000, dt=0.005, K=K, sigma=sigma, rng=np.random.default_rng(1000))
            LLE[i, j] = lle
    return Rss, LLE

def plot_heatmap(M, K_vals, S_vals, title, cbar_label, outfile):
    fig, ax = plt.subplots()
    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[min(K_vals), max(K_vals), min(S_vals), max(S_vals)],
                   cmap="viridis")
    ax.set_xlabel("Coupling K")
    ax.set_ylabel("Noise σ")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"[SAVED] {outfile}")

def write_grid_csv(M, K_vals, S_vals, path):
    with open(path, "w") as f:
        header = "sigma\\K," + ",".join([str(k) for k in K_vals]) + "\n"
        f.write(header)
        for s_val, row in zip(S_vals, M):
            f.write(str(s_val) + "," + ",".join([str(v) for v in row]) + "\n")
    print(f"[SAVED] {path}")

def main():
    pair = ensure_adjacency_csvs()
    K_vals = np.linspace(0.2, 4.0, 12)
    S_vals = np.linspace(0.0, 0.2, 9)

    for label, A in [("healthy", pair.healthy), ("perturbed", pair.perturbed)]:
        Rss, LLE = compute_grid(A, K_vals, S_vals)
        plot_heatmap(Rss, K_vals, S_vals, f"Steady-state R (mean over seeds) – {label}", "R̄_ss", 
                     os.path.join(FIG_DIR, f"heatmap_Rss_{label}.png"))
        plot_heatmap(LLE, K_vals, S_vals, f"Largest Lyapunov Exponent – {label}", "λ_max", 
                     os.path.join(FIG_DIR, f"heatmap_LLE_{label}.png"))

        write_grid_csv(Rss, K_vals, S_vals, os.path.join(TAB_DIR, f"grid_Rss_{label}.csv"))
        write_grid_csv(LLE, K_vals, S_vals, os.path.join(TAB_DIR, f"grid_LLE_{label}.csv"))

if __name__ == "__main__":
    main()