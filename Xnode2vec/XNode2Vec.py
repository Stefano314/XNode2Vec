from fastnode2vec import Node2Vec, Graph
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import networkx as nx
from skspatial.objects import Line
from scipy.spatial import distance

def generate_edgelist(df):
    """
    Description
    -----------
    Read a pandas DataFrame and generates an edge list vector to eventually build a networkx graph. The syntax of the
    file header is rigidly controlled and can't be changed. The header format must be: (node1, node2, weight).

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame edge list of the wanted network.

    Returns
    -------
    output : list
        The output of the function is a list of tuples of the form (node_1, node_2, weight).

    Note
    ----
    - In order to generate a **networkx** object it's only required to give the list to the Graph() constructor
    >>> edgelist = xn2v.generate_edgelist(DataFrame)
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from(edgelist)
    - The data types of the 'node1' and 'node2' columns must be strings, otherwise they will be converted as strings.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.array([[1, 2, 3.7], [1, 3, 0.33], [2, 7, 12]]), columns=['node1', 'node2', 'weight'])
    >>> edgelist = xn2v.generate_edgelist(df)
        [('1.0', '2.0', 3.7), ('1.0', '3.0', 0.33), ('2.0', '7.0', 12.0)]
    """
    # forcing type values
    df = df.astype({'node1': str, 'node2': str, 'weight': np.float64})
    # check header:
    header_names = list(df.columns.values)
    if header_names[0] != 'node1' or header_names[1] != 'node2' or header_names[2] != 'weight':
        raise TypeError('The header format is different from the required one.')
    return list(df.itertuples(index = False, name = None))

def edgelist_from_csv(path, **kwargs):
    """
    Description
    -----------
    Read a .csv file using pandas dataframes and generates an edge list vector to eventually build a networkx graph.
    The syntax of the file header is rigidly controlled and can't be changed.

    Parameters
    ----------
    path : string
        Path or name of the .csv file to be loaded.
    **kwargs :  pandas.read_csv() arguments

    Returns
    -------
    output : list
        The output of the function is a list of tuples of the form (node_1, node_2, weight).

    Note
    ----
    - In order to generate a **networkx** object it's only required to give the list to the Graph() constructor
    >>> edgelist = xn2v.edgelist_from_csv('some_edgelist.csv')
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from(edgelist)
    - The data types of the 'node1' and 'node2' columns must be strings, otherwise they will be converted as strings.
    
    Examples
    --------
    >>> edgelist = xn2v.edgelist_from_csv('somefile.csv')
        [('a','1',3.4),('a','2',0.6),('a','b',10)]
    """
    df_csv = pd.read_csv(path, dtype = {'node1': str, 'node2': str, 'weight': np.float64}, **kwargs)
    # check header:
    header_names = list(df_csv.columns.values)
    if header_names[0] != 'node1' or header_names[1] != 'node2' or header_names[2] != 'weight':
        raise TypeError('The header format is different from the required one.')
    return list(df_csv.itertuples(index = False, name = None))

def complete_edgelist(Z, info=False, **kwargs):
    """
        Description
        -----------
        This function performs a **data transformation** from the space points to a network. It generates links between
        specific points and gives them weights according to other conditions.

        Parameters
        ----------
        Z : numpy ndarray
            Numpy array containing as columns the i-th coordinate of the k-th point. The rows are the points, the columns
            are the coordinates.
        info :  bool
            Flag to print out some generic information of the dataset.

        Returns
        -------
        output : pandas DataFrame
            Edge list created from the given dataset expressed as a Pandas DataFrame.

        Examples
        --------
        >>> x1 = np.random.normal(7, 1, 3)
        >>> y1 = np.random.normal(9, 1, 3)
        >>> points = np.column_stack((x1, y1))
        >>> df = xn2v.complete_edgelist(points)
              node1 node2    weight
            0     0     0  0.000000
            1     0     1  1.358972
            2     0     2  2.393888
            3     1     0  1.358972
            4     1     1  0.000000
            5     1     2  1.345274
            6     2     0  2.393888
            7     2     1  1.345274
            8     2     2  0.000000
    """
    dimension = Z[0].size  # Number of coordinates per point
    NPoints = Z[:, 0].size  # Number of points
    weights = distance.cdist(Z, Z, 'euclidean') # Distance between all points
    df = pd.DataFrame(columns = ['node1', 'node2', 'weight'], **kwargs)
    l = 0
    for i in range(0, NPoints):
        for j in range(0, NPoints):
            df.loc[l] = [f"{i}", f"{j}", weights[i][j]]
            l+=1
    if info == True:
        print('\033[1m' + '--------- General Information ---------')
        print('Edge list of a fully connected network.')
        print('The weights are calculated using the euclidean norm.\n')
        print('- Space dimensionality: ', dimension)
        print('- Number of Points: ', NPoints)
        print('- Minimum weight: ', np.min(weights))
        print('- Maximum weight: ', np.max(weights))
        print('- Average weight: ', np.mean(weights))
        print('- Weight Variance: ', np.var(weights))
    return df

def stellar_edgelist(Z, info=False, **kwargs):
    """
    Description
    -----------
    This function performs a **data transformation** from the space points to a network. It generates links between
    specific points and gives them weights according to other conditions.

    Parameters
    ----------
    Z : numpy ndarray
        Numpy array containing as columns the i-th coordinate of the k-th point. The rows are the points, the columns
        are the coordinates.
    info :  bool
        Flag to print out some generic information of the dataset.

    Returns
    -------
    output : pandas DataFrame
        Edge list created from the given dataset expressed as a Pandas DataFrame.

    Examples
    --------
    >>> x1 = np.random.normal(7, 1, 6)
    >>> y1 = np.random.normal(9, 1, 6)
    >>> points_1 = np.column_stack((x1, y1))
    >>> df = xn2v.stellar_edgelist(points_1)
          node1 node2     weight
        0     0     1  12.571278
        1     0     2  11.765633
        2     0     3   9.735974
        3     0     4  12.181443
        4     0     5  11.027584
        5     0     6  12.755861

    >>> x2 = np.random.normal(107, 2, 3)
    >>> y2 = np.random.normal(101, 1, 3)
    >>> points_2 = np.column_stack((x2, y2))
    >>> tot = np.concatenate((points_1,points_2),axis=0)
    >>> df = xn2v.stellar_edgelist(tot)
          node1 node2     weight
        0     0     1  12.571278
        1     0     2  11.765633
        2     0     3   9.735974
        3     0     4  12.181443
        4     0     5  11.027584
        5     0     6  12.755861
        6     0     7  146.229997
        7     0     8  146.952899
        8     0     9  146.595700
    """
    dimension = Z[0].size # Number of coordinates per point
    NPoints = Z[:,0].size # Number of points
    weights = np.linalg.norm(Z, axis = 1)
    df = pd.DataFrame(columns = ['node1', 'node2', 'weight'], **kwargs)
    for i in range(0,NPoints):
        df.loc[i] = ['0',f"{i+1}",weights[i]]
    if info == True:
        print('\033[1m'+'--------- General Information ---------')
        print('Edge list of a stellar network.')
        print('The weights are calculated using the euclidean norm.\n')
        print('- Space dimensionality: ', dimension)
        print('- Number of Points: ', NPoints)
        print('- Minimum weight: ', np.min(weights))
        print('- Maximum weight: ', np.max(weights))
        print('- Average weight: ', np.mean(weights))
        print('- Weight Variance: ', np.var(weights))
    return df

def best_line_projection(Z):
    """
    Description
    -----------
    Performs a linear best fit of the dataset points and projects them on the line itself.

    Parameters
    ----------
    Z : numpy ndarray
        Numpy array containing as columns the i-th coordinate of the k-th point. The rows are the points, the columns
        are the coordinates.

    Returns
    -------
    output : numpy ndarray
        The output of the function is a numpy ndarray containing the transformed points of the dataset.

    Examples
    --------
    >>> x1 = np.random.normal(7, 1, 6)
    >>> y1 = np.random.normal(9, 1, 6)
    >>> points = np.column_stack((x1, y1))
    >>> xn2v.best_line_projection(points)
        [[-0.15079291  1.12774076]
         [ 2.65759595  4.44293266]
         [ 3.49319696  5.42932658]]
    """
    a = Line.best_fit(Z)
    NPoints = Z[:, 0].size
    dimension = Z[0].size
    projections = []
    for i in range(0,NPoints):
        projections.extend(np.array(a.project_point(Z[i])))
    projections = np.reshape(projections, (NPoints, dimension))
    return projections

def recover_points(Z, G, nodes):
    """
        Description
        -----------
        Recovers the spatial points from the analyzed network. It uses the fact that the order of the nodes that build
        the network is the same as the dataset one, therefore there is a one-to-one correspondence between nodes and
        points.

        Parameters
        ----------
        Z : numpy ndarray
            Numpy array containing as columns the i-th coordinate of the k-th point. The rows are the points, the columns
            are the coordinates.
        G : networkx.Graph()
            Gives the network that has been analyzed.
        nodes : list
            List of the *most similar nodes* obtained after the analysis on the network G. The type is forced to be a
            string.

        Returns
        -------
        output : numpy ndarray
            The output of the function is a numpy ndarray containing the required points from the dataset.

        Examples
        --------
        >>> x1 = np.random.normal(16, 2, 30)
        >>> y1 = np.random.normal(9, 2, 30)
        >>> x2 = np.random.normal(100, 2, 30)
        >>> y2 = np.random.normal(100, 2, 30)
        >>> family1 = np.column_stack((x1, y1)) # REQUIRED ARRAY FORMAT
        >>> family2 = np.column_stack((x2, y2)) # REQUIRED ARRAY FORMAT
        >>> dataset = np.concatenate((family1,family2),axis=0) # Generic dataset
        >>> df = xn2v.complete_edgelist(dataset) # Pandas edge list generation
        >>> df = xn2v.generate_edgelist(df)
        >>> G = nx.Graph()
        >>> G.add_weighted_edges_from(df)
        >>> nodes, similarity = xn2v.similar_nodes(G,node='1',picked=10,walk_length=20,dim=100,context=5,Weight=True)
        >>> cluseter = xn2v.recover_points(dataset,G,nodes)
        [[17.98575878  8.99318017]
         [18.03438744  9.46128979]
         [15.83803679 10.39565391]
         [15.95210332 10.4135796 ]
         [19.44550252 12.7551321 ]
         [18.62321691 10.7604561 ]
         [16.30640697 12.15702448]
         [18.73718742 13.99351914]
         [18.7817838   7.92318885]
         [16.15456589 10.72636297]]       
        """
    # check dimensionality
    if Z[:,0].size != np.array(G.nodes).size:
        print('dataset: ',Z[:,0].size)
        print('network: ', np.array(G.nodes).size)
        raise Exception('Error: the dataset dimension is different from the one of the network.')
    # force string type
    nodes = [str(s) for s in nodes]
    picked_nodes = []
    pos = 0
    for n in G:
        if n in nodes:
            picked_nodes.append(Z[pos])
        pos += 1
    return np.array(picked_nodes)


def similar_nodes(G, node=1, picked=10, train_time = 30, Weight=False, save_model = False, 
                  model_name = 'model.wordvectors' , **kwargs):
    """
    Description
    -----------
    Performs FastNode2Vec algorithm with full control on the crucial parameters.
    In particular, this function allows the user to keep working with networkx objects
    -- that are generally quite user-friendly -- instead of the ones required by the fastnode2vec
    algorithm.

    Parameters
    ----------
    G : networkx.Graph object
        Sets the network that will be analyzed by the algorithm.
    p : float
        Sets the probability '1/p' necessary to perform the fastnode2vec random walk. It affects how often the walk is
        going to immediately revisit the previous node. The smaller it is, the more likely the node will be revisited.
    q : float
        Sets the probability '1/q' necessary to perform the fastnode2vec random walk. It affects how far the walk
        will go into the network. The smaller it is, the larger will be the distance from the initial node.
    node : int, optional
        Sets the node from which to start the analysis. This is a gensim.models.word2vec parameter.
        The default value is '1'.
    walk_length : int
        Sets the number of jumps to perform from node to node.
    save_model : bool, optional
        Saves in the working directory a .wordvectors file that contains the performed training.
        It's important to consider is that the **methods** of the "Word2Vec" model saved can be accessed
        as "model_name.wv". The documentation of ".wv" is found however under 
        "gensim.models.keyedvectors.KeyedVectors" istance; they are the same thing, ".wv" is just a rename.
        The default value is 'False'.
    picked : int, optional
        Sets the first 'picked' nodes that are most similar to the node identified with 'node'. This is a
        gensim.models.word2vec parameter.
        The default value is '10'.
    train_time : int, optional
        Sets the number of times we want to apply the algorithm. It is the 'epochs' parameter in Node2Vec.
        The value of this parameter drastically affect the computational time.
        The default value is '5'.
    Weight : bool, optional
        Specifies if the algorithm must also consider the weights of the links. If the networks is unweighted this
        parameter must be 'False', otherwise it receives too many parameters to unpack.
        The default value is 'False'.
    dim : int, optional
        This is the Word2Vec "size" argument. It sets the dimension of the algorithm word vector. The longer it is, the
        more complex is the specification of the word -- object. If a subject has few features, the word length should
        be relatively short.
        The default value is '128'.
    context : int, optional
        This is the Word2Vec "window" parameter. It sets the number of words **before** and **after** the current one that will
        be kept for the analysis. Depending on its value, it manages to obtain words that are interchangeable and
        relatable -- belonging to the same topic. If the value is small, 2-15, then we will likely have interchangeable
        words, while if it is large, >15, we will have relatable words.
        The default value is '10'
    Returns
    -------
    output : ndarray, ndarray
        The output of the function is a tuple of two numpy arrays. The first contains the top 'picked' most similar
        nodes to the 'node' one, while the second contains their similarities with respect to the 'node' one.

    Notes
    -----
    - The node parameter is by default an integer. However, this only depends on the node labels that are given to the
      nodes in the network.
    - The rest of the parameters in **kwargs are the ones in fastnode2vec.Node2Vec constructor, I only specified what I
      considered to be the most important ones.
    - I noticed that the walk_length parameter should be at least #Nodes/2 in order to be a solid walk.
    
    Examples
    --------
    >>> G = nx.generators.balanced_tree(r=3, h=4)
    >>> nodes, similarity = similar_nodes(G, dim=128, walk_length=30, context=10, 
    >>>                                   p=0.1, q=0.9, workers=4)
        nodes: [0 4 5 6 45 40 14 43 13 64]
        similarity: [0.81231129 0.81083304 0.760795 0.7228986 0.66750246 0.64997339 
                     0.64365959 0.64236712 0.63170493 0.63144475]
    """
    if Weight == False:
        G_fn2v = Graph(G.edges(), directed = False, weighted = Weight)
    else:
        G_fn2v = Graph(list(G.edges.data("weight", default = 1)), directed = False, weighted = Weight)
    n2v = Node2Vec(G_fn2v, **kwargs)
    n2v.train(epochs=train_time)
    if save_model == True:
        n2v.save(model_name)
    nodes = n2v.wv.most_similar(node, topn = picked)
    nodes_id = list(list(zip(*nodes))[0])
    similarity = list(list(zip(*nodes))[1])
    nodes_id = np.array(nodes_id)
    similarity = np.array(similarity)
    return nodes_id, similarity
    
def Load(file):
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
      The documentation of ".wv" is found however under "gensim.models.keyedvectors.KeyedVectors" istance; 
      they are the same thing, ".wv" is just a rename.

    """
    model = Word2Vec.load(file)
    return model

def Draw(G, nodes_result, title = 'Community Network', **kwargs):
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
        >>> nodes, similarity = similar_nodes(G, dim=128, walk_length=30, context=100, 
        >>>                                   p=0.1, q=0.9, workers=4)
        >>> red_node = 2
        >>> nodes = np.append(nodes, red_node)
        >>> Draw(G, nodes)
    """
    color_map = []
    for node in G:
        if node == int(nodes_result[-1]):
            color_map.append('red')
        elif node in nodes_result:
            color_map.append('blue')
        else:
            color_map.append('green')
    plt.figure(figsize = (7, 5))
    ax = plt.gca()
    ax.set_title(title, fontweight = "bold", fontsize = 18, **kwargs)
    nx.draw(G, node_color = color_map, with_labels = True)
    plt.show()
