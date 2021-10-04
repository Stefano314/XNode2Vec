# XNode2Vec - An Alternative Data Clustering Procedure
Description
-----------
This repository proposes an alternative method for data classification and clustering, based on the Node2Vec algorithm that is applied to a properly transformed N-dimensional dataset.
The original [Node2Vec](https://github.com/aditya-grover/node2vec) algorithm was replaced with an extremely faster version, called [FastNode2Vec](https://github.com/louisabraham/fastnode2vec). The application of the algorithm is provided by a function that works with **networkx** objects, that are quite user-friendly. At the moment there are few easy data transformations, but they will be expanded in more complex and effective ones.

Installation
------------
In order to install the Xnode2vec package simply use pip:

- pip install Xnode2vec

*If there are some problems with the installation, please read the "Note" below.*

How to Use
----------
The idea behind is straightforward: 
1. Take a dataset, or generate one. 
2. Apply the proper transformation to the dataset.
3. Build a **networkx** object that embeds the dataset with its crucial properties.
4. Perform a node classification analysis with Node2Vec algorithm.

```
import numpy as np
import Xnode2vec as xn2v

x1 = np.random.normal(16, 1, 20)
y1 = np.random.normal(9, 1, 20)
x2 = np.random.normal(17, 2, 20)
y2 = np.random.normal(13, 1, 20)

family1 = np.column_stack((x1, y1)) # REQUIRED ARRAY FORMAT
family2 = np.column_stack((x2, y2)) # REQUIRED ARRAY FORMAT

dataset = np.concatenate((family1,family2),axis=0) # Generic dataset
transf_dataset = xn2v.best_line_projection(dataset) # Points transformation

df = xn2v.complete_edgelist(transf_dataset) # Pandas edge list generation
edgelist = xn2v.generate_edgelist(df)
G = nx.Graph()
G.add_weighted_edges_from(a) # Feed the graph with the edge list

nodes, similarity = similar_nodes(G, dim=128, walk_length=20, context=5, p=0.1, q=0.9, workers=4)
```
Note
----
9/17/2021: I had some issues when installing the fastnode2vec package; in particular, the example given by Louis Abraham gives an error. I noticed that after the installation, the declaration of the file "node2vec.py" wasn't the same as the latest version available on its GitHub (at the moment). My brutal solution was simply to just copy the whole content into the node2vec.py file. This solves the problem.

# Examples
Most Similar Nodes, Balanced Tree
---------------------------------
![tree_15](https://user-images.githubusercontent.com/79590448/132143490-64ac2417-4d21-4a87-aa42-e9e0784bcb58.png)

Most Similar Nodes Distribution, E-R
------------------------------------
![E-R_Nodes](https://user-images.githubusercontent.com/79590448/132143507-94807c17-4656-44b0-bac1-6af945d50fbf.png)

Community Network
-----------------
![Com_class](https://user-images.githubusercontent.com/79590448/134899866-713d943d-0159-40af-bda5-9297195d4596.png)

Hi-C Translocation Detection
----------------------------
![Sim3_2](https://user-images.githubusercontent.com/79590448/134982724-307334c8-74c8-48af-b6a8-88f0547fc40a.png)

