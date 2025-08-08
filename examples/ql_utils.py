# ql_utils.py
"""
Purpose:
  Shared utilities for QL brain-network Kuramoto analyses.

Inputs:
  - CSV adjacency files (float, square, symmetric):
      brain_dti_matrix_healthy.csv
      brain_dti_matrix_perturbed.csv
    If absent, synthetic surrogates are generated and saved.

Outputs (created if missing):
  - results/figures/
  - results/tables/
  - logs/

Reproducibility:
  - We expose a fixed SEED_LIST for 30 runs and document RNG use.
"""

from __future__ import annotations
import os
import json
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Tuple, Dict

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
LOG_DIR = "logs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 30 deterministic seeds covering a broad range
SEED_LIST = [
    123, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281
]

@dataclass
class AdjacencyPair:
    healthy: np.ndarray
    perturbed: np.ndarray

def _normalize_symmetric(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = np.array(A, dtype=float)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A[A < 0] = 0.0
    # Optional degree normalization to control scale
    d = A.sum(axis=1) + eps
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

def generate_synthetic_adjacencies(
    n_nodes: int = 90,
    healthy_density: float = 0.08,
    weight_scale: float = 1.0,
    global_perturb_reduction: float = 0.3,
    rng: np.random.Generator | None = None,
) -> AdjacencyPair:
    """
    Create plausible brain-like graphs:
      - Healthy: small-world backbone + modular structure + positive weights
      - Perturbed: global weight reduction + mild random deletions

    Returns matrices already normalized and symmetric.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Small-world backbone
    k = max(2, int(healthy_density * n_nodes))  # mean degree proxy
    G_ws = nx.watts_strogatz_graph(n_nodes, k if k % 2 == 0 else k + 1, 0.1, seed=42)
    # Add a modular (community) augmentation
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(G_ws.edges())

    # Add extra intra-community edges
    n_comm = 6
    community_size = n_nodes // n_comm
    for c in range(n_comm):
        nodes = list(range(c * community_size, (c + 1) * community_size))
        extra = nx.erdos_renyi_graph(len(nodes), 0.15, seed=100 + c)
        mapping = {i: nodes[i] for i in range(len(nodes))}
        extra = nx.relabel_nodes(extra, mapping)
        G.add_edges_from(extra.edges())

    # Weights
    A = np.zeros((n_nodes, n_nodes))
    for u, v in G.edges():
        base = rng.lognormal(mean=np.log(weight_scale), sigma=0.35)
        A[u, v] = base
        A[v, u] = base

    A_healthy = _normalize_symmetric(A)

    # Perturbation: global reduction + random removal of a small fraction
    A_pert = A.copy() * (1.0 - global_perturb_reduction)
    mask = rng.uniform(size=A.shape) < 0.03  # 3% random micro-lesions
    A_pert[mask] *= 0.3
    A_pert = _normalize_symmetric(A_pert)

    return AdjacencyPair(healthy=A_healthy, perturbed=A_pert)

def ensure_adjacency_csvs(
    healthy_csv: str = "brain_dti_matrix_healthy.csv",
    perturbed_csv: str = "brain_dti_matrix_perturbed.csv",
    n_nodes_default: int = 90,
) -> AdjacencyPair:
    """
    Loads CSVs if present. Otherwise generates synthetic matrices and saves them.
    """
    have_healthy = os.path.exists(healthy_csv)
    have_perturbed = os.path.exists(perturbed_csv)
    if have_healthy and have_perturbed:
        H = np.loadtxt(healthy_csv, delimiter=",")
        P = np.loadtxt(perturbed_csv, delimiter=",")
        return AdjacencyPair(healthy=_normalize_symmetric(H), perturbed=_normalize_symmetric(P))

    print("[INFO] CSVs not found; generating synthetic healthy and perturbed matrices...")
    pair = generate_synthetic_adjacencies(n_nodes=n_nodes_default)
    np.savetxt(healthy_csv, pair.healthy, delimiter=",")
    np.savetxt(perturbed_csv, pair.perturbed, delimiter=",")
    with open(os.path.join(LOG_DIR, "adjacency_generation_meta.json"), "w") as f:
        json.dump({"n_nodes": n_nodes_default}, f, indent=2)
    return pair

def spectral_measures(A: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral radius, Fiedler value, eigenvalue gap of adjacency,
    and return also the largest adjacency eigenvalue (signed).
    """
    # Adjacency spectrum by magnitude
    w, _ = np.linalg.eig(A)
    idx = np.argsort(-np.abs(w))
    w_sorted = w[idx]
    rho = float(np.max(np.abs(w_sorted)))
    gap = float(np.abs(w_sorted[0]) - np.abs(w_sorted[1])) if len(w_sorted) > 1 else float("nan")
    lambda1_signed = float(w_sorted[0].real)

    # Fiedler of Laplacian
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    mu = np.linalg.eigvalsh(L)
    mu2 = float(np.sort(mu)[1]) if len(mu) > 1 else float("nan")

    return {
        "spectral_radius": rho,
        "eigen_gap_adj": gap,
        "lambda1_adj_signed": lambda1_signed,
        "fiedler_mu2": mu2,
    }

def principal_eigenvector(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(A)
    idx = np.argmax(np.abs(w))
    vec = np.real(v[:, idx])
    vec /= np.linalg.norm(vec) + 1e-12
    return vec

def projection_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    cos2 = np.clip((np.dot(v1, v2) / ((np.linalg.norm(v1) + 1e-12) * (np.linalg.norm(v2) + 1e-12))) ** 2, 0.0, 1.0)
    return float(np.sqrt(1.0 - cos2))

def degree_and_eigencentrality(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    deg = A.sum(axis=1)
    v1 = principal_eigenvector(A)
    return deg, np.abs(v1)

def top_hubs_by(A: np.ndarray, mode: str = "degree", k: int = 5) -> np.ndarray:
    if mode == "degree":
        scores = A.sum(axis=1)
    elif mode == "eigenvector":
        scores = np.abs(principal_eigenvector(A))
    else:
        raise ValueError("mode must be 'degree' or 'eigenvector'")
    return np.argsort(-scores)[:k]