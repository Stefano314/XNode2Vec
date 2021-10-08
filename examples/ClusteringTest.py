import Xnode2vec as xn2v
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

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

nodes_families = xn2v.clusters_detection(G, cluster_rigidity = 0.75, spacing = 5, dim_fraction = 0.8, picked=100,dim=100,context=5,Weight=True, walk_length=10)
points_families = []

for i in range(0,len(nodes_families)):
    points_families.append(xn2v.recover_points(dataset,G,nodes_families[i]))

plt.scatter(dataset[:,0], dataset[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generic Dataset', fontweight='bold')
plt.show()
for i in range(0,len(nodes_families)):
    plt.scatter(points_families[i][:,0], points_families[i][:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clustered Dataset', fontweight='bold')
plt.show()