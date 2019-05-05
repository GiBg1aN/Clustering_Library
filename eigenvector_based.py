import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import pairwise_distances
from utils import gaussian_kernel, plot_clustering_result,generate_dataset


def eigenvector_thresholding(evect, threshold, clustered_elements):
    clustered_idx = []
    new_clustered_elements = clustered_elements.copy()

    for i in range(len(evect)):
        if i not in clustered_elements and abs(evect[i]) > threshold:
            clustered_idx.append(i)
            new_clustered_elements.append(i)
    return clustered_idx, new_clustered_elements


def apply_eigenvector_based(X, A, threshold=0.001):
    """
    Apply k-means to set of points X, according to input parameters.

    Args:
        X: list of 2d points
        K: number of cluster to find
        threshold: value for thresholding the characteristic vector
    
    Returns:
        clusters: vector indicating cluster for each point in X
        noise: vector of points not clustered
    """

    evals, evects = np.linalg.eig(A)
    evects = evects.T  # from column-eigenvectors to row-eigenvectors
    clusters = []
    noise = []
    clustered_elements = []
    continue_clustering = True

    while(continue_clustering):
        if np.amax(evals) == -float("inf"):
            break
        bigger_eig_id = np.argmax(evals)
        cluster, clustered_elements = eigenvector_thresholding(evects[bigger_eig_id], 
                                                               threshold, clustered_elements)
        clusters.append(cluster)
        if len(clustered_elements) == len(X):
            continue_clustering = False
        evals[bigger_eig_id] = -float("inf")
    return clusters, noise
    
