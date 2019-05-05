import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from utils import gaussian_kernel, plot_clustering_result


def eigenvector_thresholding(evect, threshold, clustered_elements):
    clustered_idx = []
    new_clustered_elements = clustered_elements.copy()

    for i in range(len(evect)):
        if i not in clustered_elements and abs(evect[i]) > threshold:
            clustered_idx.append(i)
            new_clustered_elements.append(i)
    return clustered_idx, new_clustered_elements

def main():
    # DATASET GENERATION AND PREPARATION
    np.random.seed(0)
    N_SAMPLES = 1500

    blobs = datasets.make_blobs(n_samples=N_SAMPLES, random_state=8)
    # varied = datasets.make_blobs(n_samples=N_SAMPLES, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    # noisy_moons = datasets.make_moons(n_samples=N_SAMPLES, noise=.05)
    # noisy_circles = datasets.make_circles(n_samples=N_SAMPLES, factor=.5, noise=.05)
    X, _ = blobs
    X = StandardScaler().fit_transform(X)  # normalize dataset for easier parameter selection
    D = pairwise_distances(X)  # euclidean distance as distance metric 

    mult = 0.05
    # gamma = 1 / (mult * np.var(D))
    # A = rbf_kernel(D, gamma=gamma)  # Gaussian distance as affinity metric
    A = gaussian_kernel(D, mult=mult, is_sym=True)  # Gaussian distance as affinity metric

    evals, evects = np.linalg.eig(A)
    evects = evects.T  # from column-eigenvectors to row-eigenvectors


    # CLUSTERING
    threshold = 0.001
    clusters = []
    clustered_elements = []
    continue_clustering = True

    while(continue_clustering):
        if np.amax(evals) == -float("inf"):
            break
        bigger_eig_id = np.argmax(evals)
        cluster, clustered_elements = eigenvector_thresholding(evects[bigger_eig_id], 
                                                               threshold, clustered_elements)
        clusters.append(cluster)
        if len(clustered_elements) == N_SAMPLES:
            continue_clustering = False
        evals[bigger_eig_id] = -float("inf")

    plot_clustering_result(X, A, clusters)

if __name__ == '__main__':
    main()

