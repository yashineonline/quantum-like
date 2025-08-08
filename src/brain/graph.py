# src/brain/graph.py
import networkx as nx
import numpy as np

def generate_brain_graph(n_nodes=20, k=4):
    G = nx.random_regular_graph(k, n_nodes)
    A = nx.to_numpy_array(G)
    return A, G