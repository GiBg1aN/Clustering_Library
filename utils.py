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