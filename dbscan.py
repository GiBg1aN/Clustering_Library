import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


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


def generate_cluster_matrix(A, clusters):
    res = np.empty_like(A)
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


    # PLOTTING
    colors = list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', 
                                '#999999', '#e41a1c', '#dede00']), len(clusters) + 1))
    symbols = list(islice(cycle(["*","x","+","o",".","^","<",">","P","p","D","d"]),
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
    plt.scatter(X[noise, 0], X[noise, 1], s=90, color='#000000', marker="X")
    plt.title("Clustered data (" + str(len(clusters)) + " clusters found)")

    plt.subplot(2, 2, 4)
    plt.title("Clusters affinity matrix")
    sns.heatmap(generate_cluster_matrix(A, clusters), square = True)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    main()

