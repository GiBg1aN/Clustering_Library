import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from utils import plot_clustering_result

def is_core_point(n, min_pts):
    return n >= min_pts


def extract_neighborhood(p, D, epsilon, clustered_elements):
    neighborhood = []       
    for q in range(len(D)):
        if not clustered_elements[q]:
            if D[p, q] < epsilon:
                neighborhood.append(q)
    return neighborhood


def extract_cluster(point, D, min_pts, epsilon, clustered_elements, neighbor_searching):
    cluster = []
    neighborhood = extract_neighborhood(point, D, epsilon, clustered_elements)               
    if is_core_point(len(neighborhood), min_pts) or neighbor_searching:
        cluster += neighborhood
        clustered_elements[point] = True        
        for q in neighborhood:
            clustered_elements[q] = True
        for q in neighborhood:
            if is_core_point(len(neighborhood), min_pts):
                cluster += extract_cluster(q, D, min_pts, epsilon, clustered_elements, True)
    return cluster

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

    mult = 42
    gamma = 1 / (mult * np.var(D))
    A = rbf_kernel(D, gamma=gamma)  # Gaussian distance as affinity metric


    # CLUSTERING
    epsilon = 0.2
    min_pts = 7

    clustered_elements = [False for _ in range(len(X))]
    clusters = []

    for p in range(len(X)):
        if not clustered_elements[p]:
            cluster = extract_cluster(p, D, min_pts, epsilon, clustered_elements, False)
            if cluster != []:
                clusters.append(cluster)
    noise = [i for i in range(len(clustered_elements)) if not clustered_elements[i]]

    plot_clustering_result(X, A, clusters, noise)

if __name__ == '__main__':
    main()

