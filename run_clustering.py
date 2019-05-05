import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from k_means import apply_kmeans
from dbscan import apply_dbscan
from eigenvector_based import apply_eigenvector_based
from utils import plot_clustering_result, generate_dataset, gaussian_kernel


def main():
    X = generate_dataset(shape="blobs")
    D = pairwise_distances(X)  # euclidean distance as distance metric 
    A = gaussian_kernel(D, is_sym=True)  # Gaussian distance as affinity metric
    
    # K-MEANS
    clusters, _ = apply_kmeans(X)
    plot_clustering_result(X, A, clusters, clustering_name="K means clustering")
    
    # DBSCAN
    clusters, noise = apply_dbscan(X, D)
    plot_clustering_result(X, A, clusters, noise, clustering_name="DBSCAN clustering")

    # EIGENVECTOR BASED CLUSTERING
    A_eigen = gaussian_kernel(D, mult=0.05, is_sym=True)  # Gaussian distance as affinity metric
    clusters, noise = apply_eigenvector_based(X, A_eigen)
    plot_clustering_result(X, A_eigen, clusters, noise, clustering_name="Eigenvector based clustering")


if __name__ == '__main__':
    main()