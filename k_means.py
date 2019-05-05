import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import pairwise_distances
from utils import plot_clustering_result, generate_dataset, gaussian_kernel


def compute_centroids(data, clusters, centroids):
    new_centroids = []
    for i in range(len(clusters)):
        elements = [data[j] for j in clusters[i]]
        if elements == []:
            new_centroids.append(centroids[i])
        else:
            new_centroids.append(np.mean(elements, 0))
    return new_centroids

def apply_kmeans(X, K=3, epsilon=0.0001):
    """
    Apply k-means to set of points X, according to input parameters.

    Args:
        X: list of 2d points
        K: number of cluster to find
        epsilon: value. When all the current centroids are far from
                 previous centroids less than epsilon, clsutering process
                 is stopped.
    
    Returns:
        clusters: vector indicating cluster for each point in X
        noise: vector of points not clustered
    """

    centroids = []
    continue_clustering = True
    
    for _ in range(K):  # Init random centroids
        i = np.random.choice(X.shape[0])
        centroids.append(X[i])

    while (continue_clustering):
        clusters = [[] for _ in range(K)]
        for i in range(X.shape[0]):
            distances = []
            for c in centroids:
                distances.append(np.linalg.norm(X[i] - c))
            min_cluster_idx = np.argmin(distances)
            clusters[min_cluster_idx].append(i)

        new_centroids = compute_centroids(X, clusters, centroids)
        centroids_differences = np.abs(np.array(centroids) - np.array(new_centroids))
        if np.all(centroids_differences < epsilon):
            continue_clustering = False
        centroids = new_centroids
    return clusters, None


