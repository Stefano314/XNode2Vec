import numpy as np
import pandas as pd
import networkx as nx
from Xnode2vec import best_line_projection, low_limit_network
import pytest

def test_line_points():
    """
    Checks if the number of points returned by the best_line_projection() is the same as the original dataset.
    """
    x = np.random.normal(16, 2, 100)
    y = np.random.normal(9, 2.3, 100)
    z = np.random.normal(6, 1, 100)
    dataset = np.column_stack((x, y, z))
    trans_dataset = best_line_projection(dataset)
    assert dataset.size == trans_dataset.size

def test_line_dimension():
    """
    Checks if the points returned by the best_line_projection() belong to the same space of the original dataset.
    """
    x = np.random.normal(16, 2, 100)
    y = np.random.normal(9, 2.3, 100)
    z = np.random.normal(6, 1, 100)
    dataset = np.column_stack((x, y, z))
    trans_dataset = best_line_projection(dataset)
    assert dataset[0].size == trans_dataset[0].size

def test_low_threshold():
    """
    Checks if all the values below the specified *2.13125* threshold are set to zero.
    """
    delta = 0.5
    G = nx.Graph()
    G.add_weighted_edges_from([('1', '2', 3.0), ('1', '3', 7.5), ('1', '4', 3.6), ('2', '4', 1.9),
                               ('4', '5', 1.0), ('2', '6', 5.1), ('2', '7', 1.0), ('3', '7', 11)])
    G = low_limit_network(G, delta)
    # Post-Weights
    link_weights_post = nx.get_edge_attributes(G, 'weight')
    weights_post = np.array(list(link_weights_post.values())).astype(float)
    assert np.count_nonzero(weights_post == 0) == 3