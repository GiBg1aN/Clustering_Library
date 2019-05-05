import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

def gaussian_kernel(A, mult=2, is_sym=False):
    """
    Apply gaussian kernel on matrix A.

    A(i,j) = exp(D(i,j)^2/(mult*variance(D)))

    Args:
        A: square distance matrix
        mult: kernel denominator multiplier 

    Returns:
        D: distance matrix after gaussian kernel application
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Dimensionality: input matrix is not a square matrix.')

    n = len(A)
    D = np.zeros(A.shape)
    var_D = np.var(A)
    for i in range(n):
        if is_sym:
            for j in range(i, n):
                res = np.exp(-(A[i,j] ** 2)/(mult*var_D))
                D[i,j] = res 
                D[j,i] = res 
        else:    
            for j in range(n):
                D[i,j] = np.exp(-(A[i,j] ** 2)/(mult*var_D))
    
    return D

def generate_cluster_matrix(A, clusters):
    """
    Generate the affinity matrix reordered according to clustering
    process result.

    Args:
        A: affinity matrix
        clusters: vector of cluster identifiers for each point in A
    
    Returns:
        res: reordered affinity matrix

    """
    res = np.empty_like(A)
    supp = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            supp.append(clusters[i][j])

    for i in range(len(supp)):
        for j in range(len(supp)):
            res[i, j] = A[supp[i], supp[j]]
    return res

def plot_clustering_result(X, A, clusters, noise=None):
    """
    Plot result of clustering process in a 2D space.

    Args:
        X: list of points
        A: affinity matrix between points in X
        clusters: vector of cluster identifiers for each point in X
        noise: vector of identifier for noise points

    """
    colors = list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', 
                                '#999999', '#e41a1c', '#dede00']), len(clusters) + 1))
    symbols = list(islice(cycle(["*","x","+","o",".","^","<",">","P","p","X","D","d"]),
                                len(clusters) + 1))

    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.grid()
    plt.title("Original Data")

    plt.subplot(2, 2, 2)
    plt.title("Affinity matrix")
    sns.heatmap(A, square = True)

    plt.subplot(2, 2, 3)
    plt.grid()
    for i in range(len(clusters)):
        plt.scatter(X[clusters[i], 0], X[clusters[i], 1], s=90, color=colors[i], marker=symbols[i])
    plt.title("Clustered data (" + str(len(clusters)) + " clusters found)")

    plt.subplot(2, 2, 4)
    plt.title("Clusters affinity matrix")
    sns.heatmap(generate_cluster_matrix(A, clusters), square = True)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()