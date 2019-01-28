import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel



# DATASET GENERATION AND PREPARATION
np.random.seed(0)
n_samples = 1500

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)
X, y = blobs
X = StandardScaler().fit_transform(X) # normalize dataset for easier parameter selection
D = pairwise_distances(X) # Distance matrix as euclidean distance

mult = 42
gamma = 1 / (mult * np.var(D))
A = rbf_kernel(D, gamma=gamma) # Gaussian distance function
    
# Compute eigenvalues-eigenvectors
evals, evects = np.linalg.eig(A)
evects = evects.T # from column to row vectors


# CLUSTERING
def eigenvector_thresholding(evect, threshold, clustered_elements):
    clustered_idx = []
    new_clustered_elements = clustered_elements.copy()
    for i in range(len(evect)):
        if i not in clustered_elements and abs(evect[i]) > threshold:
            clustered_idx.append(i)
            new_clustered_elements.append(i)
    return clustered_idx, new_clustered_elements


def remove_clustered(evect, clustered_idx):
    new_evect = evect.copy()
    for x in clustered_idx:
        new_evect[x] = 0 
    return new_evect


def generate_cluster_matrix(A, clusters):
    res = np.zeros_like(A)
    supp = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            supp.append(clusters[i][j])

    for i in range(len(supp)):
        for j in range(len(supp)):
            res[i, j] = A[supp[i], supp[j]]
    return res


threshold = 0.001
bigger_eig_id = 0
clusters = []
clustered_elements = []
continue_clustering = True

while(continue_clustering):
    if np.amax(evals) == -float("inf"):
        break;
    bigger_eig_id = np.argmax(evals)
    cluster, clustered_elements = eigenvector_thresholding(evects[bigger_eig_id], threshold, clustered_elements)
    clusters.append(cluster)
    if len(clustered_elements) == n_samples:
        continue_clustering = False
    evals[bigger_eig_id] = -float("inf")

print(len(clusters), "clusters found")


# PLOTTING
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                                    '#984ea3', '#999999', '#e41a1c', '#dede00']),
                              int(len(clusters)+ 1))))
symbols = np.array(list(islice(cycle(["*","x","+","o",".","^","<",">","P","p","X","D","d"]),
                              len(clusters) + 1)))

plt.figure(0)
plt.subplot(2, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.grid()
plt.title("Original Data")

G = nx.Graph()
for node in range(len(X)):
    G.add_node(node)

G.add_edges_from(nx.from_numpy_array(A).edges())
plt.subplot(2, 3, 2)
nx.draw_networkx(G, [(x, y) for x, y in X], node_size=15, with_labels=False)
plt.grid()
plt.title("Graph Data")

plt.subplot(2, 3, 3)
plt.title("Affinity matrix")
sns.heatmap(A, square = True, cbar_kws={"shrink": .2})

plt.subplot(2, 3, 4)
plt.grid()
for i in range(len(clusters)):
    plt.scatter(X[clusters[i], 0], X[clusters[i], 1], s=90, color=colors[i], marker=symbols[i])
plt.title("Clustered data")

plt.subplot(2, 3, 5)
plt.scatter(list(range(len(evals))), evals, s=10)
plt.grid()
plt.title("Eigenvalues")

plt.subplot(2, 3, 6)
plt.title("Clusters affinity matrix")
sns.heatmap(generate_cluster_matrix(A, clusters), square = True, cbar_kws={"shrink": .2})

plt.show()

