"""
Brain Graph Example
==================
This example generates and visualizes a synthetic brain graph.
"""
from brain.graph import generate_brain_graph
import matplotlib.pyplot as plt
import networkx as nx

A, G = generate_brain_graph(n_nodes=20, k=4)
nx.draw(G, with_labels=True)
plt.show()

