import numpy as np
import scipy.spatial.distance as distance
from numpy.fft import fft, ifft

def random_argmin(A):
    # Choose random argmin instead of smallest index to prevent identical centroids
    return np.array([np.random.choice(np.where(A[i] == A[i].min())[0]) for i in range(len(A))])

def calc_dist(a, b):
    # Distance calculation using Eucildean distance between aligned (by maximal cross correlation) points
    return distance.cdist(a, b, lambda u, v: np.sqrt(((np.roll(u, np.argmax(ifft(fft(u).conj()
                                                                                         * fft(v)).real))-v)**2).sum()))

def pp_init(y, K):
    # Choose one centroid at random
    idx = np.random.choice(len(y), 1, replace=False)
    centroids = y[idx, :]

    for _ in range(1, K):
        dist = np.min(calc_dist(y, centroids), axis=1)
        prob = np.power(dist, 2) / np.sum(np.power(dist, 2))
        idx = np.random.choice(len(y), 1, replace=False, p=prob)
        centroids = np.append(centroids, y[idx, :], axis=0)

    return centroids


def kmeans(y,K,max_iterations=100):
    # Initialize centroids using kmeans++ method
    centroids = pp_init(y, K)
    # Calculate distance between pairs of points and centroids, choose minimal distance index as the label of the point
    P = random_argmin(calc_dist(y, centroids))

    for _ in range(max_iterations):
        # Recalculate centroids by template matching (align all points in cluster with the first point, then compute mean of cluster)
        centroids = [np.array([np.roll(y[P==i,:][n], np.argmax(ifft(fft(y[P==i,:][n]).conj()
                    * fft(y[P==i,:][0])).real)) for n in range(len(y[P==i,:]))]).mean(axis=0) for i in np.unique(P)]
        tmp = random_argmin(calc_dist(y, centroids))
        if np.array_equal(P,tmp):break
        P = tmp
    return P