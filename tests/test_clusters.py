import numpy as np
import pandas as pd
import networkx as nx
from Xnode2vec import cluster_generation, clusters_detection, similar_nodes, recover_points
import pytest

def test_cluster_zero_threshold():
    """
    Description
    -----------
    Test of cluster_generation() function.
    Checks if giving a 0 similarity threshold value will give back the whole vector.
    """
    nodes = np.array(['1','2','3',4,5])
    similarities = np.random.rand(nodes.size)
    result = [nodes, similarities]
    cluster = cluster_generation(result, cluster_rigidity = 0.)
    assert np.array_equal(nodes,cluster)

def test_cluster_one_threshold():
    """
    Description
    -----------
    Test of cluster_generation() function.
    Checks if giving a 1 similarity threshold value will give back an empty vector.
    """
    nodes = np.array(['1','2','3',4,5])
    similarities = np.random.rand(nodes.size)
    result = [nodes, similarities]
    cluster = cluster_generation(result, cluster_rigidity = 1.)
    assert cluster.size == 0

def test_similar_nodes_dimensions():
    """
    Description
    -----------
    Test of similar_nodes() function.
    Checks if the dimensions of nodes and similarities are the same.
    """
    r = np.random.randint(1,6)
    h = np.random.randint(1,6)
    G = nx.generators.balanced_tree(r=r, h=h)
    rand_node = np.random.randint(0, len(list(G.nodes)))
    nodes, similarities = similar_nodes(G,node=rand_node,context=5,dim=100,walk_length=int(len(list(G.nodes))/2))
    assert nodes.size == similarities.size

def test_similar_nodes_picked():
    """
    Description
    -----------
    Test of similar_nodes() function.
    Checks if the 'picked' parameter returns a vector of dimension 'picked'.
    """
    G = nx.generators.balanced_tree(r=3, h=3)
    rand_picked = np.random.randint(0,10)
    nodes, similarities = similar_nodes(G, node=1, picked = rand_picked, context=5, dim=40, walk_length=5)
    assert nodes.size == rand_picked

def test_similar_nodes_community():
    """
    Description
    -----------
    Test of similar_nodes() function.
    Checks if the analysis identifies the correct family of the selected node.
    """
    r = np.random.randint(4, 8)
    G = nx.balanced_tree(r, 2)
    G.remove_node(0)
    nodes, similarities = similar_nodes(G, node = 2, picked = r, context = 5, dim = 100, walk_length = 15)
    comparison = np.sort(nodes) == np.array([n for n in G.neighbors(2)])
    assert comparison.all()

def test_recover_order():
    """
    Description
    -----------
    Test of recover_points() function.
    Checks if the order between the given dataset and its network representation is maintained.
    """
    dataset = np.array([[1,2,3,4], #point1
                        [7.2,3,4.1,9], #point2
                        [6,7,8,9], #point3
                        [11,4.4,5,6.2], #point4
                        [0.9,3.2,5,18.2], #point5
                        [24,1.1,5.9,6], #point6
                        [1,4.4,8,0.2]]) #point7
    G = nx.Graph()
    G.add_weighted_edges_from([('point1','point2',3.0),('point1','point3',7.5),('point1','point4',3.6),('point2','point4',1.9),
                               ('point4','point5',1.0),('point2','point6',5.1),('point2','point7',1.0),('point3','point7',11)])
    picked_nodes = ['point1','point5','point7']
    expected_points = np.array([[1,2,3,4],
                                [0.9,3.2,5,18.2],
                                [1,4.4,8,0.2]])
    cluster = recover_points(dataset, G, picked_nodes)
    assert np.array_equal(cluster, expected_points)

def test_recover_picked_nodes_permutation():
    """
    Description
    -----------
    Test of recover_points() function.
    Checks if a generic permutation of the picked nodes affects the dataset recover points. This is clearly crucial,
    since the sorting order of the picked nodes is generally different when performing a different simulation on the
    same dataset.
    """
    dataset = np.array([[1,2,3,4], #point1
                        [7.2,3,4.1,9], #point2
                        [6,7,8,9], #point3
                        [11,4.4,5,6.2], #point4
                        [0.9,3.2,5,18.2], #point5
                        [24,1.1,5.9,6], #point6
                        [1,4.4,8,0.2]]) #point7
    G = nx.Graph()
    G.add_weighted_edges_from([('point1','point2',3.0),('point1','point3',7.5),('point1','point4',3.6),('point2','point4',1.9),
                               ('point4','point5',1.0),('point2','point6',5.1),('point2','point7',1.0),('point3','point7',11)])
    picked_nodes = ['point1','point5','point7']
    permuted_nodes = np.random.permutation(picked_nodes)
    cluster = recover_points(dataset, G, picked_nodes)
    permuted_cluster = recover_points(dataset, G, permuted_nodes)
    assert np.array_equal(cluster,permuted_cluster)

def test_clusters_dimension1():
    """
    Description
    -----------
    Test of clusters_detection() function.
    Tests if the *dim* parameter value is dim = 0 then the cluster won't be expanded.
    """
    G = nx.generators.balanced_tree(100,1)
    nodes_families, unlabeled_nodes = clusters_detection(G, cluster_rigidity=0.9, spacing=30, dim_fraction=0.,
                                                         picked=int(G.number_of_nodes()/10), dim=100,context=5, walk_length=30)
    assert len(nodes_families)>=1

def test_clusters_dimension2():
    """
    Description
    -----------
    Test of clusters_detection() function.
    Tests if the *dim* parameter value is dim > 1 then only one cluster will be generated.
    """
    G = nx.generators.balanced_tree(100,1)
    nodes_families, unlabeled_nodes = clusters_detection(G, cluster_rigidity=0.7, spacing=30, dim_fraction=1.01,
                                                         picked=int(G.number_of_nodes()), dim=100,context=5, walk_length=30)
    assert len(nodes_families)==1

def test_clusters_rigidity1():
    """
    Description
    -----------
    Test of clusters_detection() function.
    Tests if the *cluster_rigidity* parameter value is cluster_rigidity = 0 then the cluster will contain all the nodes
    in the network.
    """
    G = nx.generators.balanced_tree(100,1)
    nodes_families, unlabeled_nodes = clusters_detection(G, cluster_rigidity=0., spacing=50, dim_fraction=1.,
                                                         picked=G.number_of_nodes(), dim=100,context=5, walk_length=30)
    assert nodes_families[0].size == 101
