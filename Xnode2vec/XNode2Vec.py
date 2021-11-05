from fastnode2vec import Node2Vec, Graph
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import networkx as nx
from skspatial.objects import Line
from scipy.spatial import distance

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
    # check header:
    header_names = list(df.columns.values)
    if header_names[0] != 'node1' or header_names[1] != 'node2' or header_names[2] != 'weight':
        raise TypeError('The header format is different from the required one.')
    # forcing type values
    df = df.astype({'node1': str, 'node2': str, 'weight': np.float64})
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

def complete_edgelist(Z, metric='euclidean', stretch=1., info=False, **kwargs):
    """
        Description
        -----------
        This function performs a **data transformation** from the space points to a network. It generates links between
        specific points and gives them weights according to the specified metric.

        Parameters
        ----------
        Z : numpy ndarray
            Numpy array containing as columns the i-th coordinate of the k-th point. The rows are the points, the columns
            are the coordinates.
        metric : string, optional
            Specifies the metric in which the dataset Z is defined. The metric will determine the values of the weights
            between the links.
        stretch : float, optional
            Enlarges the distance between different points exponentially.
            The default value is '1.0'.
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
            0     0     0  1.000000
            1     0     1  0.015445
            2     0     2  0.018235
            3     1     0  0.015445
            4     1     1  1.000000
            5     1     2  0.834821
            6     2     0  0.018235
            7     2     1  0.834821
            8     2     2  1.000000
    """
    dimension = Z[0].size  # Number of coordinates per point
    NPoints = Z[:, 0].size  # Number of points
    weights = stretch*np.exp(-stretch*distance.cdist(Z, Z, metric))  # Distance between all points
    weights = weights.flatten() # Weights coulumn
    nodes_id = np.arange(NPoints).astype(str)
    node1 = np.repeat(nodes_id,NPoints)
    node2 = np.tile(nodes_id,NPoints)
    df = pd.DataFrame({'node1': node1, 'node2': node2, 'weight': weights}, **kwargs)
    if info == True:
        print('\033[1m' + '--------- General Information ---------')
        print('Edge list of a fully connected network.')
        print('The weights are calculated using minus the exponential of the norm defined by the chosen metric.\n')
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
    specific points and gives them weights according to specific conditions.

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
          node1      node2     weight
        0     origin     0  12.571278
        1     origin     1  11.765633
        2     origin     2   9.735974
        3     origin     3  12.181443
        4     origin     4  11.027584
        5     origin     5  12.755861
    >>> x2 = np.random.normal(107, 2, 3)
    >>> y2 = np.random.normal(101, 1, 3)
    >>> points_2 = np.column_stack((x2, y2))
    >>> tot = np.concatenate((points_1,points_2),axis=0)
    >>> df = xn2v.stellar_edgelist(tot)
          node1      node2     weight
        0     origin     0  12.571278
        1     origin     1  11.765633
        2     origin     2   9.735974
        3     origin     3  12.181443
        4     origin     4  11.027584
        5     origin     5  12.755861
        6     origin     6  146.229997
        7     origin     7  146.952899
        8     origin     8  146.595700
    """
    dimension = Z[0].size  # Number of coordinates per point
    NPoints = Z[:, 0].size  # Number of points
    dimension = Z[0].size  # Number of coordinates per point
    NPoints = Z[:, 0].size  # Number of points
    weights = np.exp(-np.linalg.norm(Z, axis = 1))
    node2 = np.arange(NPoints).astype(str)
    df = pd.DataFrame({'node1': 'origin', 'node2': node2, 'weight': weights}, **kwargs)
    if info == True:
        print('\033[1m' + '--------- General Information ---------')
        print('Edge list of a stellar network.')
        print('The weights are calculated using minus the exponential of the euclidean norm.\n')
        print('- Space dimensionality: ', dimension)
        print('- Number of Points: ', NPoints)
        print('- Minimum weight: ', np.min(weights))
        print('- Maximum weight: ', np.max(weights))
        print('- Average weight: ', np.mean(weights))
        print('- Weight Variance: ', np.var(weights))
    return df

def low_limit_network(G, delta, remove=False):
    """
    Description
    -----------
    This function performs a **network transformation**. It sets the link weights of the network to 0 if their initial
    value was below a given threshold. The threshold is chosen to be a constant times the average links weight.
    
    Parameters
    ----------
    G : networkx.Graph()
        Gives the network that will be modified.
    delta :  float
        Set the multiplying constant of the maximum link weight that will define the weight threshold. This number
        should be between 0 and 1.

    Returns
    -------
    output : networkx.Graph()
        Returns the networkx graph() resulting after the transformation.
    
    Examples
    --------
    >>> print(G)
    Graph with 300 nodes and 45150 edges
    >>> G = xn2v.low_limit_network(G, 0.5, remove=True)
    >>> print(G)
    Graph with 300 nodes and 14702 edges
    """
    link_weights = nx.get_edge_attributes(G, 'weight')
    weights = np.array(list(link_weights.values())).astype(float)
    max_weight = np.amax(weights)
    if remove==True:
        to_remove = []
        for u, v, d in G.edges(data = True):
            if d['weight'] <= delta * max_weight:
                to_remove.append((u,v))
        G.remove_edges_from(to_remove)
    else:
        for u, v, d in G.edges(data = True):
            if d['weight'] <= delta * max_weight:
                d['weight'] = 0.
    return G

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
    for i in range(0, NPoints):
        projections.extend(np.array(a.project_point(Z[i])))
    projections = np.reshape(projections, (NPoints, dimension))
    return projections

def cluster_generation(result, cluster_rigidity = 0.7):
    """
        Description
        -----------
        This function takes the nodes that have a similarity higher than the one set by *cluster_rigidity*.

        Parameters
        ----------
        result : list
            This parameter must be a list of two elements: the first is the nodes labels vector, the other is their
            similarities vector.
        cluster_rigidity : float, optional
            Sets the similarity threshold of the nodes. It should be a number between 0 and 1. The default value is
            '0.7'.

        Returns
        -------
        output : numpy ndarray
            The output is a numpy array containing the nodes that satisfy the condition required.

        Examples
        --------
        >>> nodes = np.arange(0,5)
            [0 1 2 3 4]
        >>> similarities = [0.5,0.9,0.91,0.87,0.67]
        >>> xn2v.cluster_generation([nodes,similarities], cluster_rigidity = 0.75)
            ['1' '2' '3']
    """
    cluster = np.array(result[0],dtype = str)
    positions = np.where(np.array(result[1]) >= cluster_rigidity)
    return(cluster[positions])

def clusters_detection(G, cluster_rigidity=0.7, spacing=5, dim_fraction=0.8, **kwargs):
    """
        Description
        -----------
        This function detects the **clusters** that compose a generic dataset. The dataset must be given as a
        **networkx** graph, using the proper data transformation. The clustering procedure uses Node2Vec algorithm to
        find the most similar nodes in the network.

        Parameters
        ----------
        G : networkx.Graph()
            Gives the network that will be analyzed.
        cluster_rigidity : float, optional
            Sets the similarity threshold of the nodes. It should be a number between 0 and 1. The default value is
            '0.7'.
        spacing : int, optional
            Sets the increment of the position when picking the nodes for the analysis. The nodes are picked every
            *spacing* value between the initial node label and the last one. The default value is '5'.
        dim_fraction : float, optional
            Sets the minumum threshold for choosing when to expand the previously generated cluster. This parameter
            appears when considering the current most similar nodes picked by the Node2Vec algorithm; in particular
            it sets the minumum number of nodes that are already present in the generated clusters family before adding
            the remaining ones to the previous cluster.

        Note
        ----
        - The function returns only the nodes labels from a specific network. In order to go back to the initial points
          it's necessary to use the function **xn2v.recover_points(dataset,Graph,Clusters[i])**. Up to now, you can
          insert only a one-dimensional vector inside **xn2v.recover_points()**, meaning that if you have N clusters
          inside the list obtained by **xn2v.clusters_detection()** you'll have to call **xn2v.recover_points()**
          N times.

        Returns
        -------
        output : list
            The output is list of numpy arrays containing nodes labels from the given network *G*. Each array represents
            a specific cluster.

        Examples
        --------
        >>> df = xn2v.edgelist_from_csv('somefile.csv')
        >>> df = xn2v.generate_edgelist(df)
        >>> xn2v.clusters_detection(G, cluster_rigidity = 0.75, spacing = 5, dim_fraction = 0.8, picked=100,
        >>>                         dim=100,context=5,Weight=True, walk_length=20)
            --------- Clusters Information ---------
            - Number of Clusters:  3
            - Total nodes:  61
            - Clustered nodes:  52
            - Number of unlabeled nodes:  9
            [array(['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '2', '3', '4', '5', '6', '7', '9'], dtype='<U2'),
            array(['21', '23', '24', '26', '27', '28', '29', '30', '31', '32', '33',
            '34', '35', '36', '37', '38', '39'], dtype='<U2'),
            array(['41', '42', '44', '45', '47', '48', '49', '50', '51', '52', '53',
            '54', '55', '56', '57', '58', '59'], dtype='<U2')]
    """
    clusters = []
    nodes_clustered = []
    index = 0
    for node_id in list(G.nodes)[::spacing]:
        flag = True
        nodes, similarities = similar_nodes(G, node_id, **kwargs)
        current_cluster = cluster_generation([nodes, similarities], cluster_rigidity) # TESTING THIS OBJECT.
        dimension = np.size(current_cluster)
        different_nodes_counters = [] # This is used to find the cluster to expand and to decide whether to create a new cluster.
        if len(clusters) != 0:
            for previous_cluster in clusters:
                # True = already in, False = new node. This is inverted, so True = new node.
                current_positions = np.invert(np.isin(current_cluster, previous_cluster))
                different_nodes_counters.append(current_positions.sum())
                current_cluster = current_cluster[~np.in1d(current_cluster,previous_cluster)] # current_cluster filtering.
        if dimension == np.size(current_cluster):
            # Creating new cluster if the dimension of cluster remain the same. This means that the nodes in common are none.
            if dimension == 0: # Stop only if the first cluster is empty.
                if index == 0: raise Exception("Error: The dimension of the cluster is 0. Try to reduce cluster_rigidity.")
                else:
                    warnings.warn("Warning: The dimension of the cluster is 0. You may want to reduce cluster_rigidity.",RuntimeWarning)
                    flag = False
            if flag == True:
                print("- Creating new cluster")
                clusters.append(current_cluster)
                index += 1
        elif dimension - np.size(current_cluster) < int(dimension * dim_fraction):
            # - If all nodes are different, dimension-np.size(current_cluster)==0 => Create new cluster.
            # - If all nodes are the same, dimension-np.size(current_cluster)==dimension => Expand with nothing.
            # - If some nodes are different, dimension-np.size(current_cluster)>0:
            #       if dimension-np.size(current_cluster)<threshold => There are enough SAME nodes to expand the clusters.
            ind = np.where(different_nodes_counters == np.min(different_nodes_counters))
            if ind[0].size != 1: # Don't expand if there are ambiguities on where put the nodes.
                pass
            else:
                # We extend the cluster that had the highest number of nodes in common. The current_cluster array is
                # already filtered with the different nodes in the whole clusters list.
                print(f"- Expand cluster number: {ind[0][0]+1}")
                clusters[ind[0][0]] = np.append(clusters[ind[0][0]], current_cluster)
        else:
            pass
    tot_nodes = [val for sublist in clusters for val in sublist]  # Flatten list of lists
    unlabeled = list(set(np.array(list(G.nodes)).astype(str)) - set(tot_nodes))  # Get remaining nodes
    print('\033[1m' + '--------- Clusters Information ---------')
    print('- Number of Clusters: ', index)
    print('- Total nodes: ', len(tot_nodes) + len(unlabeled))
    print('- Clustered nodes: ', len(tot_nodes))
    print('- Number of unlabeled nodes: ', len(unlabeled))
    for i in range(0,len(clusters)):
        print(f"- Nodes in cluster {i+1}:", clusters[i].size)
    return clusters, unlabeled

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
    if np.array(G.nodes)[0] == 'origin':
        if Z[:,0].size+1 != np.array(G.nodes).size:
            raise Exception(f"Error: the dataset dimension dim={Z[:, 0].size} is different from the one expected for the network dim={np.array(G.nodes).size}.")
        else: Z = np.insert(Z, 0, np.zeros(Z[0].size), axis=0) # Adding origin
    elif Z[:,0].size != np.array(G.nodes).size:
            raise Exception(f"Error: the dataset dimension dim={Z[:, 0].size} is different from the one of the network dim={np.array(G.nodes).size}.")
    # force string type
    nodes = [str(s) for s in nodes]
    picked_nodes = []
    pos = 0
    for n in G:
        if n in nodes:
            picked_nodes.append(Z[pos])
        pos += 1
    return np.array(picked_nodes)

def similar_nodes(G, node=1, picked=10, Epochs = 10, Weight=False, save_model = False,
                  model_name = 'model.wordvectors', graph = None, **kwargs):
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
    Epochs : int, optional
        Sets the number of walks that start from the given node. It is strongly suggested to use more than one
        walk.
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
        This is the Word2Vec `window` parameter. Depending on its value, it manages to obtain words that are
        interchangeable and relatable -- belonging to the same topic. If the value is small, 2-15, then we will likely
        have interchangeable words, while if it is large, >15, we will have relatable words.
        It is namely the maximum distance between the current and predicted word within a `sentence`.
        The default value is '10'

    Returns
    -------
    output : ndarray, ndarray
        The output of the function is a tuple of two numpy arrays. The first contains the top 'picked' most similar
        nodes to the 'node' one, while the second contains their similarities with respect to the 'node' one. The
        similarity is calculated as the `cosine similarity` between the vectors relative to the words themselves.
        Clearly, the dimension of the word vectors is given by `dim` parameter.

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
    >>> nodes, similarity = xn2v.similar_nodes(G, dim=128, walk_length=30, context=10, 
    >>>                                   p=0.1, q=0.9, workers=4)
        nodes: [0 4 5 6 45 40 14 43 13 64]
        similarity: [0.81231129 0.81083304 0.760795 0.7228986 0.66750246 0.64997339 
                     0.64365959 0.64236712 0.63170493 0.63144475]
    """
    if graph == None:
        G_fn2v = nx_to_Graph(G, Weight)
        n2v = Node2Vec(G_fn2v, **kwargs)
        n2v.train(epochs=Epochs)
        if save_model == True:
            n2v.save(model_name)
        nodes = n2v.wv.most_similar(node, topn = picked)
        nodes_id = list(list(zip(*nodes))[0])
        similarity = list(list(zip(*nodes))[1])
        nodes_id = np.array(nodes_id)
        similarity = np.array(similarity)
    else:
        n2v = Node2Vec(graph, **kwargs)
        n2v.train(epochs = Epochs)
        if save_model == True:
            n2v.save(model_name)
        nodes = n2v.wv.most_similar(node, topn = picked)
        nodes_id = list(list(zip(*nodes))[0])
        similarity = list(list(zip(*nodes))[1])
        nodes_id = np.array(nodes_id)
        similarity = np.array(similarity)
    return nodes_id, similarity
    
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
      The documentation of ".wv" is found however under "gensim.models.keyedvectors.KeyedVectors" istance; 
      they are the same thing, ".wv" is just a rename.

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
