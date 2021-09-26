import XNode2Vec as fn2v
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import networkx as nx

# Families threshold function
def Families(result, family_condition = 0.7):
    result = np.r_['1,2,0', result[0], result[1]]
    family_array = [] #rows are families
    for i in range(0,result[:,0].size): #most similar nodes
        if result[i,1] >= family_condition: family_array.append(int(result[i,0]))
        else: break
    return np.array(family_array)

# Distance Matrix Plot Function
def distance_plot(A=[], title='Distance Matrix', txt='none'):
    if txt == 'none':
        plt.matshow(A, cmap = 'Reds', norm = col.LogNorm())
    else:
        A = np.loadtxt(open(txt, 'rb'),
                       delimiter = ',')
        plt.matshow(A, cmap = 'Reds', norm = col.LogNorm())
    plt.title(title, fontweight = 'bold', size = 17, pad = 10)
    plt.xlabel("Node Label")
    plt.ylabel("Node Label")
    plt.colorbar()
    plt.show()

# Load Contact Map
A = np.loadtxt(open("cancer_hic.txt", 'rb'),delimiter = ',')
#A = A[250:577,250:577] # For Chr6-ChrX Translocation.
#A = A[577:,577:] # For Chr10-Chr20 Translocation.
G = nx.from_numpy_matrix(A)
#distance_plot(A)

#============== FAMILY CLASSIFICATION ================
#From Hi-C:
#chr1: 0,249
#chr6: 250,421
#chrX: 422,577
#chr10: 578,713
#chr20: 714,777

starting_node = 1
walk_l = int(G.number_of_nodes()/4)
print(walk_l)
p, q = 1, 10
families = []

# ============================ GLOBAL CONTACT MAP ===========================
# k=0
# labels = ['Chr1','Chr6','ChrX','Chr10','Chr20']
# for node_id in [3,252,430,585,716]:
#     plt.hist(Families( fn2v.n2v_algorithm(G, node_id, picked = 700, Weight = True, dim = 80, walk_length = walk_l, context = 5, p = p,
#                            q = q, workers = 4, train_time = 5), 0.56),bins=np.linspace(0, 777, 770),label = labels[k])
#     k += 1
# ============================================================================

for node_id in range(0,55,2):
    families.append(Families(fn2v.n2v_algorithm(G,node_id,picked=700,Weight=True,dim=80,walk_length=walk_l,context=5,p=p,q=q,workers=4,train_time=5),
                             0.4))
X = [item for sublist in families for item in sublist]
plt.hist(families, bins=90)
plt.title('Similar Nodes, Families', fontweight="bold", fontsize=18)
plt.xlabel('Nodes ID',  fontsize=16)
plt.ylabel('Nodes Picked',  fontsize=16)
# plt.legend()
plt.show()
# ===================================================