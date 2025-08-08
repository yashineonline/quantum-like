# kuramoto.py
"""
Core Kuramoto simulator and diagnostics.

Functions:
  - simulate_kuramoto
  - compute_order_parameter
  - estimate_lyapunov
  - unwrap_time_series

Notes:
  - Uses Euler–Maruyama discretization.
  - Noise term uses shared RNG; pass 'rng' to control reproducibility.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

def wrap_pi(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))

def compute_order_parameter(theta_t: np.ndarray) -> np.ndarray:
    """
    theta_t: array [T, N]
    returns R_t: [T]
    """
    return np.abs(np.mean(np.exp(1j * theta_t), axis=1))

def simulate_kuramoto(
    A: np.ndarray,
    T: int = 1000,
    dt: float = 0.01,
    K: float = 2.0,
    sigma: float = 0.0,
    omega_mean: float = 0.0,
    omega_std: float = 0.1,
    theta0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
      'theta': [T, N] phases, wrapped to (-pi, pi]
      'R':     [T] order parameter magnitude
      'omega': [N] intrinsic frequencies
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = A.shape[0]
    if theta0 is None:
        theta = rng.uniform(-np.pi, np.pi, size=N)
    else:
        theta = np.array(theta0, dtype=float)

    omega = rng.normal(omega_mean, omega_std, size=N)
    theta_hist = np.zeros((T, N), dtype=float)
    for t in range(T):
        coupling = K * (A @ np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])).diagonal()
        noise = np.sqrt(2.0 * max(sigma, 0.0) * dt) * rng.normal(0.0, 1.0, size=N)
        theta = wrap_pi(theta + dt * (omega + coupling) + noise)
        theta_hist[t] = theta

    R_t = compute_order_parameter(theta_hist)
    return {"theta": theta_hist, "R": R_t, "omega": omega}

def unwrap_time_series(theta_t: np.ndarray) -> np.ndarray:
    """
    Unwrap phases over time per node.
    Input [T, N] -> output [T, N]
    """
    return np.unwrap(theta_t, axis=0)

def estimate_lyapunov(
    A: np.ndarray,
    T: int = 5000,
    dt: float = 0.005,
    K: float = 2.0,
    sigma: float = 0.0,
    omega: np.ndarray | None = None,
    eps: float = 1e-6,
    renorm_every: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Benettin method with identical noise in both trajectories.
    Returns approximate largest Lyapunov exponent (per unit time).
    """
    if rng is None:
        rng = np.random.default_rng(1234)
    N = A.shape[0]
    base_rng = rng
    theta = base_rng.uniform(-np.pi, np.pi, size=N)
    if omega is None:
        omega = base_rng.normal(0.0, 0.1, size=N)

    # Nearby trajectory
    v = base_rng.normal(0.0, 1.0, size=N)
    v = v / (np.linalg.norm(v) + 1e-12) * eps
    theta_p = wrap_pi(theta + v)

    sum_logs = 0.0
    steps = 0

    for t in range(T):
        # Shared noise for both
        noise = np.sqrt(2.0 * max(sigma, 0.0) * dt) * base_rng.normal(0.0, 1.0, size=N)

        def step(x):
            coupling = K * (A @ np.sin(x[np.newaxis, :] - x[:, np.newaxis])).diagonal()
            return wrap_pi(x + dt * (omega + coupling) + noise)

        theta = step(theta)
        theta_p = step(theta_p)

        if (t + 1) % renorm_every == 0:
            diff = wrap_pi(theta_p - theta)
            dist = np.linalg.norm(diff)
            if dist < 1e-20:  # avoid log(0)
                # random re-seeding of separation
                diff = base_rng.normal(0.0, 1.0, size=N)
                diff = diff / (np.linalg.norm(diff) + 1e-12) * eps
                theta_p = wrap_pi(theta + diff)
                continue
            sum_logs += np.log(dist / eps)
            steps += 1
            # Renormalize separation
            diff = diff / dist * eps
            theta_p = wrap_pi(theta + diff)

    if steps == 0:
        return float("nan")
    return (sum_logs / steps) / (renorm_every * dt)