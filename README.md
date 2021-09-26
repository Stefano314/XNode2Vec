# XNode2Vec
Description
-----------
This repository is meant to show the strength of Node2Vec prediction algorithm and to apply it to very different types of networks. In addition, the original [Node2Vec](https://github.com/aditya-grover/node2vec) algorithm was replaced with an extremely faster version, called [FastNode2Vec](https://github.com/louisabraham/fastnode2vec). The application of the algorithm is provided by a function that works with **networkx** objects, that are quite user-friendly.

Note
-----------
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
![Families](https://user-images.githubusercontent.com/79590448/133808870-0c045f12-60d5-49d2-971f-e39c16aa1852.png)
