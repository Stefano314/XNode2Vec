[![PyPI
version](https://badge.fury.io/py/Xnode2vec.svg)](https://badge.fury.io/py/Xnode2vec)


# XNode2Vec - An Alternative Data Clustering Procedure
Description
-----------
This repository proposes an alternative method for data classification and clustering, based on the Node2Vec algorithm that is applied to a properly transformed N-dimensional dataset.
The original [Node2Vec](https://github.com/aditya-grover/node2vec) algorithm was replaced with an extremely faster version, called [FastNode2Vec](https://github.com/louisabraham/fastnode2vec). The application of the algorithm is provided by a function that works with **networkx** objects, that are quite user-friendly. At the moment there are few easy data transformations, but they will be expanded in more complex and effective ones.

Installation
------------
In order to install the Xnode2vec package simply use pip:

- ``` pip install Xnode2vec ```

*If there are some problems with the installation, please read the "Note" below.*

How to Use
----------
The idea behind is straightforward: 
1. Take a dataset, or generate one. 
2. Apply the proper transformation to the dataset.
3. Build a **networkx** object that embeds the dataset with its crucial properties.
4. Perform a node classification analysis with Node2Vec algorithm.

```python
import numpy as np
import Xnode2vec as xn2v
import pandas as pd

x1 = np.random.normal(4, 1, 20)
y1 = np.random.normal(5, 1, 20)
x2 = np.random.normal(17, 2, 20)
y2 = np.random.normal(13, 1, 20)

family1 = np.column_stack((x1, y1)) # REQUIRED ARRAY FORMAT
family2 = np.column_stack((x2, y2)) # REQUIRED ARRAY FORMAT

dataset = np.concatenate((family1,family2),axis=0) # Generic dataset
transf_dataset = xn2v.best_line_projection(dataset) # Points transformation

df = xn2v.complete_edgelist(transf_dataset) # Pandas edge list generation
edgelist = xn2v.generate_edgelist(df)
G = nx.Graph()
G.add_weighted_edges_from(edgelist) # Feed the graph with the edge list

nodes, similarity = xn2v.similar_nodes(G, dim=128, walk_length=20, context=5, picked=10, p=0.1, q=0.9, workers=4)

similar_points = xn2v.recover_points(dataset,G,nodes) # Final cluster
```
Using the same setup as before, let's perform an even more complex analysis:

```python
x1 = np.random.normal(16, 2, 100)
y1 = np.random.normal(9, 2, 100)
x2 = np.random.normal(25, 2, 100)
y2 = np.random.normal(25, 2, 100)
x3 = np.random.normal(2, 2, 100)
y3 = np.random.normal(1, 2, 100)
x4 = np.random.normal(30, 2, 100)
y4 = np.random.normal(70, 2, 100)

family1 = np.column_stack((x1, y1)) # REQUIRED ARRAY FORMAT
family2 = np.column_stack((x2, y2)) # REQUIRED ARRAY FORMAT
family3 = np.column_stack((x3, y3)) # REQUIRED ARRAY FORMAT
family4 = np.column_stack((x4, y4)) # REQUIRED ARRAY FORMAT
dataset = np.concatenate((family1,family2,family3,family4),axis=0) # Generic dataset

df = xn2v.complete_edgelist(dataset) # Pandas edge list generation
df = xn2v.generate_edgelist(df) # Networkx edgelist format
G = nx.Graph()
G.add_weighted_edges_from(df)
nodes_families, unlabeled_nodes = xn2v.clusters_detection(G, cluster_rigidity = 0.85, spacing = 15, dim_fraction = 0.8,
                                                          picked=100,dim=100,context=5,Weight=True, walk_length=20)
points_families = []
points_unlabeled = []

for i in range(0,len(nodes_families)):
    points_families.append(xn2v.recover_points(dataset,G,nodes_families[i]))
points_unlabeled = xn2v.recover_points(dataset,G,unlabeled_nodes)

plt.scatter(dataset[:,0], dataset[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generic Dataset', fontweight='bold')
plt.show()
```
Now the list ```points_families``` contains the four clusters -- clearly taking in account possible statistical errors. The results are however surprisingly good in many situations.

Objects Syntax
--------------
Here we report the list of structures required to use the Xnode2vec package:
- Dataset: ``` dataset = np.array([[1,2,3,..], ..., [1,2,3,..]])```; the rows corresponds to each point, while the coulumns to the coordinates.
- Edge List: ``` edgelist = [(node_a,node_b,weight), ... , (node_c,node_d,weight)] ```; this is a list of tuples, structured as [starting_node, arriving_node, weight]
- DataFrame: ``` pandas.DataFrame(np.array([[1, 2, 3.7], ..., [2, 7, 12]]), columns=['node1', 'node2', 'weight']) ```

Note
----
- 9/17/2021: I had some issues when installing the fastnode2vec package; in particular, the example given by Louis Abraham gives an error. I noticed that after the installation, the declaration of the file "node2vec.py" wasn't the same as the latest version available on its GitHub (at the moment). My brutal solution was simply to just copy the whole content into the node2vec.py file. This solves the problem.
- 9/17/2021: Numba requires numpy <= 1.20 in order to work.

# Examples
Most Similar Nodes, Balanced Tree
---------------------------------
<img src="https://user-images.githubusercontent.com/79590448/132143490-64ac2417-4d21-4a87-aa42-e9e0784bcb58.png" width="550">

Most Similar Nodes Distribution, E-R
------------------------------------
<img src="https://user-images.githubusercontent.com/79590448/132143507-94807c17-4656-44b0-bac1-6af945d50fbf.png" width="550">

Community Network
-----------------
<img src="https://user-images.githubusercontent.com/79590448/134899866-713d943d-0159-40af-bda5-9297195d4596.png" width="550">

Hi-C Translocation Detection
----------------------------
<img src="https://user-images.githubusercontent.com/79590448/134982724-307334c8-74c8-48af-b6a8-88f0547fc40a.png" width="550">

Clustering Test
---------------
<img src="https://user-images.githubusercontent.com/79590448/136523320-ca10d879-ce69-4216-af2c-d7fc674b17cd.png" width="420">  <img src="https://user-images.githubusercontent.com/79590448/137181948-85c9b21a-52cd-4493-b6b9-35f9bcb5c4da.png" width="420">

