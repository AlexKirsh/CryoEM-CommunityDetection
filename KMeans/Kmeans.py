import numpy as np
import scipy.spatial.distance as distance
from numpy.fft import fft, ifft

def random_argmin(A):
    # Choose random argmin instead of smallest index to prevent identical centroids
    return np.array([np.random.choice(np.where(A[i] == A[i].min())[0]) for i in range(len(A))])

def kmeans(y,K,max_iterations=100):
    idx = np.random.choice(len(y), K, replace=False)
    centroids = y[idx, :] # Choose K random point from dataset
    # Calculate distance between pairs of points and centroids, choose minimal distance index as the label of the point
    P = random_argmin(distance.cdist(y, centroids, lambda u, v: np.sqrt(((np.roll(u, np.argmax(ifft(fft(u).conj()
                                     * fft(v)).real))-v)**2).sum())))

    for _ in range(max_iterations):
        # Recalculate centroids by template matching (align all points in cluster with the first point, then compute mean of cluster)
        centroids = [np.array([np.roll(y[P==i,:][n], np.argmax(ifft(fft(y[P==i,:][n]).conj()
                    * fft(y[P==i,:][0])).real)) for n in range(len(y[P==i,:]))]).mean(axis=0) for i in range(K)]
        tmp = random_argmin(distance.cdist(y, centroids, lambda u, v: np.sqrt(((np.roll(u, np.argmax(ifft(fft(u).conj()
                                           * fft(v)).real))-v)**2).sum())))
        if np.array_equal(P,tmp):break
        P = tmp
    return P