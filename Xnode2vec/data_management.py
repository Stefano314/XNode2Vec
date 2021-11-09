from fastnode2vec import Node2Vec, Graph
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def nx_to_Graph(G, Weight = False):
    """
    Description
    -----------
    Performs a conversion from the **networkx** graph format to the **fastnode2vec** one, that is necessary to work with
    fastnode2vec objects.
    
    Parameters
    ----------
    G : networkx.Graph()
        Gives the network that will be converted.
    Weight : bool
        Specifies if the network is a weighted one.
        
    Returns
    -------
    output : fastnode2vec.Graph
        The output of the function is a **fastnode2vec** Graph object.
    """
    if Weight == False:
        G_fn2v = Graph(G.edges(), directed = False, weighted = Weight)
    else:
        G_fn2v = Graph(list(G.edges.data("weight", default = 1)), directed = False, weighted = Weight)
    return G_fn2v

def labels_modifier(G, new_ids):
    """
    Description
    -----------
    Changes the labels of the created networkx graph. It can be useful if we want to select rows from a dataframe that
    we can't recover only with their positions in the vector.
    
    Parameters
    ----------
    G : networkx.Graph()
        Gives the network that will be modified.
    new_ids : list
        Ordered list of the new node labels.
        
    Returns
    -------
    output : networkx.Graph()
        Returns the same networkx graph() with the new labels.
        
    Note
    ----
    - The number of nodes and edges of the original network won't change. Only specific weight values will be set to 0.
    
    Examples
    --------
    >>> G = nx.generators.balanced_tree(3,2)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    >>> new_indexes = ['node1',2,'noise','x',4,5,'main_hub',8,9,10,11,12,'end_point']
    >>> G = xn2v.labels_modifier(G,new_indexes)
    ['node1', 2, 'noise', 'x', 4, 5, 'main_hub', 8, 9, 10, 11, 12, 'end_point']
    """
    mapping = dict(zip(G, new_ids))
    new_G = nx.relabel_nodes(G, mapping)
    return new_G

def load_model(file):
    """
    Parameters
    ----------
    file : .wordvectors
        Gives file name of the saved word2vec model to load a "gensim.models.keyedvectors.KeyedVectors"
        object.
        
    Returns
    -------
    model : Word2Vec object.
        This is the previously saved model.
        
    Note
    ----
    - I put this function just to compress everything useful for an analysis, without having to 
      call the gensim method.
    
    - It's important to consider is that the **methods** of the "Word2Vec" model saved can be accessed as "model_name.wv". 
      The documentation of ".wv" is found however under "gensim.models.keyedvectors.KeyedVectors" istance.
    """
    model = Word2Vec.load(file)
    return model

def draw_community(G, nodes_result, title='Community Network', **kwargs):
    """
    Description
    -----------
    Draws a networkx plot highlighting some specific nodes in that network. The last node is higlighted in red, the
    remaining nodes in "nodes_result" are in blue, while the rest of the network is green.
    
    Parameters
    ----------
    G : networkx.Graph object
        Sets the network that will be drawn.
    nodes_result : ndarray
        Gives the nodes that will be highlighted in the network. The last element will be red, the others blue.
    title : string, optional
        Sets the title of the plot.
        
    Notes
    -----
    - This function returns a networkx draw plot, which is good only for networks with few nodes (~40). For larger
      networks I suggest to use other visualization methods, like Gephi.
      
    Examples
    --------
    >>> G = nx.generators.balanced_tree(r=3, h=4)
    >>> nodes, similarity = xn2v.similar_nodes(G, dim=128, walk_length=30, context=100, 
    >>>                                   p=0.1, q=0.9, workers=4)
    >>> red_node = 2
    >>> nodes = np.append(nodes, red_node)
    >>> xn2v.draw_community(G, nodes)
    """
    color_map = []
    for node in G:
        if str(node) == str(nodes_result[-1]):
            color_map.append('red')
        elif str(node) in nodes_result.astype(str):
            color_map.append('blue')
        else:
            color_map.append('green')
    plt.figure(figsize = (7, 5))
    ax = plt.gca()
    ax.set_title(title, fontweight = "bold", fontsize = 18, **kwargs)
    nx.draw(G, node_color = color_map, with_labels = True)
    plt.show()
