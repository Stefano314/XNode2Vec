import numpy as np
import networkx as nx
from Xnode2vec import best_line_projection, low_limit_network, nx_to_Graph
from fastnode2vec import Graph
import pytest

def test_line_points():
    """
    Description
    -----------
    Test of best_line_projection() function.
    Checks if the transformed dataset has the same number of points and lies in the same space as the original.
    """
    x = np.random.normal(16, 2, 100)
    y = np.random.normal(9, 2.3, 100)
    z = np.random.normal(6, 1, 100)
    dataset = np.column_stack((x, y, z))
    trans_dataset = best_line_projection(dataset)
    assert dataset.shape == trans_dataset.shape

def test_low_threshold_zeros():
    """
    Description
    -----------
    Test of low_limit_network() function.
    Checks if all the values below the specified *2.09* threshold are set to zero.
    """
    delta = 0.19
    G = nx.Graph()
    G.add_weighted_edges_from([('1', '2', 3.0), ('1', '3', 7.5), ('1', '4', 3.6), ('2', '4', 1.9),
                               ('4', '5', 1.0), ('2', '6', 5.1), ('2', '7', 1.0), ('3', '7', 11)])
    G = low_limit_network(G, delta, remove=False)
    # Post-Weights
    link_weights_post = nx.get_edge_attributes(G, 'weight')
    weights_post = np.array(list(link_weights_post.values())).astype(float)
    assert np.count_nonzero(weights_post == 0) == 3

def test_low_threshold_remove():
    """
    Description
    -----------
    Test of low_limit_network() function.
    Checks if all the values below the specified *2.09* threshold are set to zero.
    """
    delta = 0.19
    G = nx.Graph()
    G.add_weighted_edges_from([('1', '2', 3.0), ('1', '3', 7.5), ('1', '4', 3.6), ('2', '4', 1.9),
                               ('4', '5', 1.0), ('2', '6', 5.1), ('2', '7', 1.0), ('3', '7', 11)])
    G = low_limit_network(G, delta, remove=True)
    assert not any([G.has_edge('2','4'), G.has_edge('4','5'), G.has_edge('2','7')])

def test_nx_to_Graph():
    """
    Description
    -----------
    Test of nx_to_Graph() function.
    Checks if the Graph created by nx_to_Graph() is the same as the one required.
    """
    edgelist = [('point1', 'point2', 3.0), ('point1', 'point3', 7.5), ('point1', 'point4', 3.6),
                ('point2', 'point4', 1.9), ('point2', 'point6', 5.1), ('point2', 'point7', 1.0), 
                ('point3', 'point7', 11), ('point4', 'point5', 1.0)]
    graph = Graph(edgelist, directed = False, weighted = True) # Expected Graph
    G = nx.Graph()
    G.add_weighted_edges_from(edgelist)
    graph_nx = nx_to_Graph(G, Weight = True)
    assert np.array_equal(graph_nx.data, graph.data)

def test_summary_edgelist1():
    """
    Description
    -----------
    Test of summary_edgelist() function, for complete_edgelist().
    Checks if the output printed is the expected one.
    """
    dataset = np.array([[1, 2, 3, 11],
                        [0.5, 9, 2.4, 9],
                        [5, 3, 2, 7]])
    df = complete_edgelist(dataset)
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    summary_edgelist(dataset,df)
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    assert output == """\x1b[1m--------- General Information ---------
Edge list of a fully connected network.
- Space dimensionality:  4
- Number of Points:  3
- Minimum weight:  0.0
- Maximum weight:  0.002935281570570076
- Average weight:  0.0008927563147689381
- Weight Variance:  1.2541199053237623e-06\n"""

def test_summary_edgelist2():
    """
    Description
    -----------
    Test of summary_edgelist() function, for stellar_edgelist().
    Checks if the output printed is the expected one.
    """
    dataset = np.array([[1, 2, 3, 11],
                        [0.5, 9, 2.4, 9],
                        [5, 3, 2, 7]])
    df = stellar_edgelist(dataset)
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    summary_edgelist(dataset,df)
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    assert output == """\x1b[1m--------- General Information ---------
Edge list of a stellar network.
- Space dimensionality:  4
- Number of Points:  3
- Minimum weight:  2.3481866887962857e-06
- Maximum weight:  8.895507836838607e-05
- Average weight:  3.3432430173058003e-05
- Weight Variance:  1.548743426810968e-09\n"""

def test_summary_clusters():
    """
    Description
    -----------
    Test of summary_clusters() function.
    Checks if the output printed is the expected one.
    """
    cl1 = np.array(['a', 'b', '4'], dtype = object)
    cl2 = np.array(['3', 'y', 'x', '11'], dtype = object)
    clustered = [cl1, cl2]
    unclustered = np.array(['2', '7', 'n', 'c'], dtype = object)
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    summary_clusters(clustered, unclustered)
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    assert output=="""\033[1m--------- Clusters Information ---------
- Number of Clusters: 2
- Total nodes: 11
- Clustered nodes:  7
- Number of unlabeled nodes: 4
- Nodes in cluster 1: 3
- Nodes in cluster 2: 4\n"""
