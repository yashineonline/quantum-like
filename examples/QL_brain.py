# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# brain_graph.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic brain graph (can be replaced with actual DTI adjacency matrix)
def generate_brain_graph(n_nodes=20, k=4):
    G = nx.random_regular_graph(k, n_nodes)
    A = nx.to_numpy_array(G)
    return A, G


# %%
A, G = generate_brain_graph()
nx.draw(G, with_labels=True)  # Visualize the graph
plt.show()

# %%
# spectral_analysis.py
from scipy.linalg import eig

def dominant_eigenmode(A):
    eigvals, eigvecs = eig(A)
    idx = np.argmax(np.real(eigvals))
    return np.real(eigvals), np.real(eigvecs[:, idx])


# %%
# kuramoto_simulation.py
import numpy as np

def kuramoto_step(theta, omega, A, K, dt):
    return theta + dt * (omega + K * np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1))

def simulate_kuramoto(A, steps=1000, dt=0.01, K=1.0):
    N = A.shape[0]
    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.normal(0, 1, N)
    history = [theta.copy()]
    for _ in range(steps):
        theta = kuramoto_step(theta, omega, A, K, dt)
        history.append(theta.copy())
    return np.array(history)


# %%
# perturbation.py
def weaken_edges(A, reduction_factor=0.3):
    A_perturbed = A.copy()
    A_perturbed *= (1 - reduction_factor)
    np.fill_diagonal(A_perturbed, 0)  # No self loops
    return A_perturbed


# %%
# plotting.py
import matplotlib.pyplot as plt

def plot_phase_history(history):
    plt.figure(figsize=(10, 6))
    for i in range(history.shape[1]):
        plt.plot(history[:, i], alpha=0.6)
    plt.title("Kuramoto Phase Evolution")
    plt.xlabel("Time step")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.show()



# %%
# graph_features.py
def extract_graph_features(A):
    eigvals = np.linalg.eigvalsh(A)
    return {
        'spectral_radius': np.max(eigvals),
        'fiedler_value': np.sort(eigvals)[1],
        'zero_eigen_count': np.sum(np.isclose(eigvals, 0.0))
    }


# %%
# run_pipeline.py
#from brain_graph import generate_brain_graph
#from spectral_analysis import dominant_eigenmode
#from kuramoto_simulation import simulate_kuramoto
#from perturbation import weaken_edges

# Step 1: Generate graph
A, G = generate_brain_graph()

# Step 2: Spectral analysis
eigvals, eigvec = dominant_eigenmode(A)

# Step 3: Simulate Kuramoto
history_clean = simulate_kuramoto(A)

# Step 4: Perturb and re-run
A_perturbed = weaken_edges(A)
history_perturbed = simulate_kuramoto(A_perturbed)

# %%
plot_phase_history(history_clean)

# %%
plot_phase_history(history_perturbed)

# %%
