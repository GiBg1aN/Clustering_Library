import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from itertools import cycle, islice
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


def eigenvector_thresholding(evect, threshold, clustered_elements):
    clustered_idx = []
    new_clustered_elements = clustered_elements.copy()

    for i in range(len(evect)):
        if i not in clustered_elements and abs(evect[i]) > threshold:
            clustered_idx.append(i)
            new_clustered_elements.append(i)
    return clustered_idx, new_clustered_elements


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
        
    evals, evects = np.linalg.eig(A)
    evects = evects.T  # from column-eigenvectors to row-eigenvectors


    # CLUSTERING
    threshold = 0.001
    clusters = []
    clustered_elements = []
    continue_clustering = True

    while(continue_clustering):
        if np.amax(evals) == -float("inf"):
            break;
        bigger_eig_id = np.argmax(evals)
        cluster, clustered_elements = eigenvector_thresholding(evects[bigger_eig_id], 
                                                               threshold, clustered_elements)
        clusters.append(cluster)
        if len(clustered_elements) == N_SAMPLES:
            continue_clustering = False
        evals[bigger_eig_id] = -float("inf")


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
    plt.title("Clustered data (" + str(len(clusters)) + " clusters found)")

    plt.subplot(2, 2, 4)
    plt.title("Clusters affinity matrix")
    sns.heatmap(generate_cluster_matrix(A, clusters), square = True)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    main()

