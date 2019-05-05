import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import pairwise_distances
from utils import plot_clustering_result, generate_dataset, gaussian_kernel

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

def apply_dbscan(X, D, epsilon=0.2, min_pts=7):
    """
    Apply dbscan to set of points X, according to input parameters.

    Args:
        X: list of 2d points
        D: pairwise distance matrix
        epsilon: radius for epsilon neighbourhood of a point
        min_pts: minimum number of elements such that a point has high density

    Returns:
        clusters: vector indicating cluster for each point in X
        noise: vector of points not clustered
    """

    clustered_elements = [False for _ in range(len(X))]
    clusters = []

    for p in range(len(X)):
        if not clustered_elements[p]:
            cluster = extract_cluster(p, D, min_pts, epsilon, clustered_elements, False)
            if cluster != []:
                clusters.append(cluster)
    noise = [i for i in range(len(clustered_elements)) if not clustered_elements[i]]
    return clusters, noise

