import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


def compute_centroids(data, clusters):
    new_centroids = []
    for cluster in clusters:
        elements = [data[i] for i in cluster]
        new_centroids.append(np.mean(elements, 0))
    return new_centroids


def euclidean_dist(x, y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


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


def main():
    # DATASET GENERATION AND PREPARATION
    np.random.seed(0)
    N_SAMPLES = 1500

    blobs = datasets.make_blobs(n_samples=N_SAMPLES, random_state=8)
    # varied = datasets.make_blobs(n_samples=N_SAMPLES, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    X, _ = blobs
    X = StandardScaler().fit_transform(X)  # normalize dataset for easier parameter selection
    D = pairwise_distances(X)  # euclidean distance as distance metric 

    mult = 42
    gamma = 1 / (mult * np.var(D))
    A = rbf_kernel(D, gamma=gamma)  # Gaussian distance as affinity metric


    # CLUSTERING
    K = 3
    epsilon = 0.0001
    clusters = []
    centroids = []
    continue_clustering = True
        
    for _ in range(K):  # Init random centroids
        i = np.random.choice(X.shape[0])
        centroids.append(X[i])
        clusters.append([])

    while (continue_clustering):
        clusters = [[] for _ in range(K)]
        for i in range(X.shape[0]):
            distances = []
            for c in centroids:
                distances.append(euclidean_dist(X[i], c))
            min_cluster_idx = np.argmin(distances)
            clusters[min_cluster_idx].append(i)

        new_centroids = compute_centroids(X, clusters)
        centroids_differences = np.abs(np.array(centroids) - np.array(new_centroids))
        if np.all(centroids_differences < epsilon):
            continue_clustering = False
        centroids = new_centroids


    # PLOTTING
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
    plt.title("Clustered data")

    plt.subplot(2, 2, 4)
    plt.title("Clusters affinity matrix")
    sns.heatmap(generate_cluster_matrix(A, clusters), square = True)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    main()
