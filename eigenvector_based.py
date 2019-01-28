import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
from sklearn import datasets, cluster
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice


np.random.seed(0)

# GENERATE DATASET
n_samples = 1500
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

X, y = blobs
# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

# use 10-nn to generate adjacency matrix
# A = kneighbors_graph(X, n_neighbors=10, mode='distance', include_self=False)
A = radius_neighbors_graph(X, 0.3, mode='distance', include_self=False)

# PLOTTING
plt.figure(0)
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.grid()

G = nx.Graph()
for node in range(len(X)):
    G.add_node(node)

G.add_edges_from(nx.from_scipy_sparse_matrix(A).edges())
plt.subplot(2, 2, 2)
nx.draw_networkx(G, [(x, y) for x, y in X], node_size=15, with_labels=False)
plt.grid()


# COMPUTE EIGENVALUES-EIGENVECTORS
A = A.toarray()

plt.subplot(2, 2, 3)
sns.heatmap(np.corrcoef(A), square = True)

evals, evects = np.linalg.eig(A)

idx = evals.argsort()[::-1]   
evals = evals[idx]
evects = evects[:,idx]


# CLUSTERING
threshold = 0.001
bigger_eig_id = 0
clusters = []
continue_clustering = True


def eigenvector_thresholding(evect, threshold):
    clustered_idx = []
    for i in range(len(evect)):
        if evect[i] > threshold:
            clustered_idx.append(i)
    return clustered_idx

def remove_clustered(evect, clustered_idx):
    new_evect = evect.copy()
    for x in clustered_idx:
        new_evect[x] = 0 
    return new_evect

while(continue_clustering):
    clusters.append(eigenvector_thresholding(evects[bigger_eig_id], threshold))
    if clusters[bigger_eig_id] == []:
        continue_clustering = False
    else:
        bigger_eig_id += 1
        for clustered_idx in clusters:
            evects[bigger_eig_id] = remove_clustered(evects[bigger_eig_id], clustered_idx)

print(bigger_eig_id, "clusters found")

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(bigger_eig_id + 1))))
plt.subplot(2, 2, 4)
plt.grid()
for i in range(len(clusters)):
    plt.scatter(X[clusters[i], 0], X[clusters[i], 1], s=10, color=colors[i])
plt.show()
