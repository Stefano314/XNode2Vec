import numpy as np
import pandas as pd
import networkx as nx
from Xnode2vec import cluster_generation, clusters_detection, similar_nodes, recover_points
import pytest

def test_cluster_zero_threshold():
    """
    Checks if giving a 0 similarity threshold value will give back the whole vector.
    """
    nodes = np.array(['1','2','3',4,5])
    similarities = np.random.rand(nodes.size)
    result = [nodes, similarities]
    cluster = cluster_generation(result, cluster_rigidity = 0.)
    assert np.array_equal(nodes,cluster)

def test_cluster_one_threshold():
    """
    Checks if giving a 1 similarity threshold value will give back an empty vector.
    """
    nodes = np.array(['1','2','3',4,5])
    similarities = np.random.rand(nodes.size)
    result = [nodes, similarities]
    cluster = cluster_generation(result, cluster_rigidity = 1.)
    assert cluster.size == 0

def test_similar_nodes_dimensions():
    """
    Checks if the dimensions of nodes and similarities are the same.
    """
    r = np.random.randint(1,6)
    h = np.random.randint(1,6)
    G = nx.generators.balanced_tree(r=r, h=h)
    rand_node = np.random.randint(0, len(list(G.nodes)))
    nodes, similarities = similar_nodes(G,node=rand_node,context=5,dim=100,walk_length=int(len(list(G.nodes))/2))
    assert nodes.size == similarities.size

def test_similar_nodes_community():
    """
    Checks if the analysis identifies the correct family of the selected node.
    WARNING: Node2Vec can travel also between two unconnected nodes, I don't know why yet, but it can also be useful.
    """
    r = np.random.randint(4, 8)
    G = nx.balanced_tree(r, 2)
    G.remove_node(0)
    nodes, similarities = similar_nodes(G, node = 2, picked = r, context = 5, dim = 100, walk_length = 15)
    comparison = np.sort(nodes) == np.array([n for n in G.neighbors(2)])
    assert comparison.all()