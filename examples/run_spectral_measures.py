# run_spectral_measures.py
"""
Compute spectral measures for healthy and perturbed adjacency:
  - spectral radius ρ(A)
  - Fiedler value μ2(L)
  - adjacency eigenvalue gap Δ
  - principal eigenvector projection distance

Outputs:
  - results/tables/spectral_measures.csv
  - Console printout
"""

import os
import numpy as np
from ql_utils import ensure_adjacency_csvs, spectral_measures, principal_eigenvector, projection_distance, TAB_DIR

def main():
    pair = ensure_adjacency_csvs()
    H, P = pair.healthy, pair.perturbed

    sm_H = spectral_measures(H)
    sm_P = spectral_measures(P)
    vH = principal_eigenvector(H)
    vP = principal_eigenvector(P)
    dproj = projection_distance(vH, vP)

    header = ["condition", "spectral_radius", "eigen_gap_adj", "lambda1_adj_signed", "fiedler_mu2"]
    rows = [
        ["healthy", sm_H["spectral_radius"], sm_H["eigen_gap_adj"], sm_H["lambda1_adj_signed"], sm_H["fiedler_mu2"]],
        ["perturbed", sm_P["spectral_radius"], sm_P["eigen_gap_adj"], sm_P["lambda1_adj_signed"], sm_P["fiedler_mu2"]],
        ["Principal eigenvector projection_distance_v1", dproj, np.nan, np.nan, np.nan],
    ]

    out = os.path.join(TAB_DIR, "spectral_measures.csv")
    with open(out, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")
    print(f"[SAVED] {out}")

    print("[SPECTRAL SUMMARY]")
    print("  Healthy:", sm_H)
    print("  Perturbed:", sm_P)
    print(f"  Principal eigenvector projection distance: {dproj:.4f}")

if __name__ == "__main__":
    main()