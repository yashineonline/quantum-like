# run_order_parameter.py
"""
Compute and plot global order parameter R(t) with 95% CI bands across 30 seeds
for healthy and perturbed networks. Also perform Wilcoxon rank-sum test on
final R distributions and annotate on the figure.

Usage:
  python run_order_parameter.py

Inputs:
  - brain_dti_matrix_healthy.csv
  - brain_dti_matrix_perturbed.csv
    If absent, they are synthesized and saved automatically.

Outputs:
  - results/figures/order_parameter_Rt.png
  - results/tables/final_R_summary.csv
  - Console summary statistics (means, CIs, p-value)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from ql_utils import ensure_adjacency_csvs, SEED_LIST, FIG_DIR, TAB_DIR
from kuramoto import simulate_kuramoto

# Style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

def mean_ci(x: np.ndarray, axis=0, alpha=0.05):
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1)
    n = x.shape[axis]
    z = 1.96  # ~95% CI
    half = z * s / np.sqrt(n)
    return m, half

def main():
    pair = ensure_adjacency_csvs()

    T = 1000    # time steps
    dt = 0.01
    K = 2.0
    sigma = 0.05

    def run_condition(A):
        R_runs = []
        for seed in SEED_LIST:
            rng = np.random.default_rng(seed)
            out = simulate_kuramoto(A, T=T, dt=dt, K=K, sigma=sigma, rng=rng)
            R_runs.append(out["R"])
        return np.vstack(R_runs)  # [S, T]

    R_healthy = run_condition(pair.healthy)
    R_pert    = run_condition(pair.perturbed)

    # Final R distributions
    Rh_final = R_healthy[:, -1]
    Rp_final = R_pert[:, -1]

    # Stats
    m_h, ci_h = mean_ci(R_healthy, axis=0)
    m_p, ci_p = mean_ci(R_pert, axis=0)

    U, pval = mannwhitneyu(Rh_final, Rp_final, alternative="two-sided")

    # Plot
    t = np.arange(R_healthy.shape[1]) * dt
    fig, ax = plt.subplots()
    ax.plot(t, m_h, color="#1f77b4", label="Healthy (mean)")
    ax.fill_between(t, m_h - ci_h, m_h + ci_h, color="#1f77b4", alpha=0.25, label="Healthy 95% CI")
    ax.plot(t, m_p, color="#d62728", label="Perturbed (mean)")
    ax.fill_between(t, m_p - ci_p, m_p + ci_p, color="#d62728", alpha=0.25, label="Perturbed 95% CI")
    ax.set_title("Global Synchronization R(t)")
    ax.set_xlabel("Time (s)  [dt=0.01]")
    ax.set_ylabel("Order parameter R")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    ax.text(0.02, 0.06, f"Mann–Whitney U p={pval:.2e}", transform=ax.transAxes)

    out_path = os.path.join(FIG_DIR, "order_parameter_Rt.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] {out_path}")

    # Table
    rows = []
    rows.append(["condition", "mean_final_R", "std_final_R", "n_seeds"])
    rows.append(["healthy", float(np.mean(Rh_final)), float(np.std(Rh_final, ddof=1)), len(Rh_final)])
    rows.append(["perturbed", float(np.mean(Rp_final)), float(np.std(Rp_final, ddof=1)), len(Rp_final)])
    rows.append(["p_value_mannwhitney", pval, np.nan, np.nan])

    tab_path = os.path.join(TAB_DIR, "final_R_summary.csv")
    with open(tab_path, "w") as f:
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")
    print(f"[SAVED] {tab_path}")

    print("[SUMMARY]")
    print(f"  Healthy final R:  mean={np.mean(Rh_final):.3f}, std={np.std(Rh_final, ddof=1):.3f}")
    print(f"  Perturbed final R: mean={np.mean(Rp_final):.3f}, std={np.std(Rp_final, ddof=1):.3f}")
    print(f"  Mann–Whitney U p-value: {pval:.3e}")

if __name__ == "__main__":
    main()