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

"""
Brain Graph Example
===================

This example generates and visualizes a synthetic brain graph using a random
regular graph. The adjacency matrix and the graph object are returned for
further analysis.
"""
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
