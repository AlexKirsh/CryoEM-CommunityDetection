import networkx as nx
import igraph as ig
import numpy as np
from scipy import stats
from numpy.fft import fft, ifft
from itertools import combinations

def generate_graph(weights_list, combs, node_labels):
    nx_G = nx.Graph()
    ig_G = ig.Graph()
    # Add nodes
    for i in range(len(node_labels)):
        nx_G.add_node(i, label=node_labels[i])
        ig_G.add_vertex(i)

    # Add edges
    for i in range(len(weights_list)):
        nx_G.add_edge(combs[:, 0][i], combs[:, 1][i], weight=weights_list[i])

    ig_G.add_edges(list(zip(combs[:, 0], combs[:, 1])))
    ig_G.es['weight'] = weights_list
    return nx_G, ig_G

# General MRA formula: y_i[n] = R_(s_i){x_(k_i)[n]} + \sigma \epsilon_i[n], \forall i \in [1,...,N]
# 1D MRA formula: y_i[n] = x_(k_i)[<n+s_i>_L] + \sigma \epsilon_i[n], \forall i \in [1,...,N]
# N - number of MRA samples, K - number of signals, L - signal length, sigma - noise parameter, x- signals
def generate_MRA(N, K, L, sigma, x):
    k = stats.randint.rvs(low=0, high=K, size=N)  # Random uniformly distributed selections of signals
    s = stats.randint.rvs(low=0, high=L, size=N)  # Random uniformly distributed selections of shifts

    # Generate Noise array
    epsilon = np.zeros((N, L))
    for n in range(N):
        epsilon[n] = sigma * np.random.randn(L)

    # Generate MRA samples
    y = np.zeros((N, L))
    true_signals = np.zeros(N)  # List the holds the signal from which y[i] sample was generated, where i is the index
    for n in range(N):
        true_signals[n] = k[n]
        shifted_x = np.roll(x[k[n]], int(s[n]))
        y[n] = shifted_x + epsilon[n]
        #y[n] = (y[n] - np.mean(y[n])) / np.linalg.norm(y[n] - np.mean(y[n]), 2)  # Normalize signal
        #y[n] = (y[n] - np.mean(y[n])) / np.std(y[n])  # Normalize signal

    return y, true_signals, epsilon


def generate_maxcorr(N, L, y, threshold=0):
    combs = np.array(list(combinations(range(len(y)),2)))
    max_corr = np.max(ifft(fft(y[combs[:, 0], :]).conj() * fft(y[combs[:, 1], :])).real, axis=1)
    max_corr = np.where(max_corr < 0, 0, max_corr) # Negative correlation presumed as no correlation

    return max_corr, combs
