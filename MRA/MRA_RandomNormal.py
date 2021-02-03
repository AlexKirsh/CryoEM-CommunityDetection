import numpy as np
from scipy import stats
import Generate
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
N = 100 # Number of observations
L = 50 # Signals length
K = 10 # Number of signals
sigma = 0.1 # Noise level

x = np.zeros((K,L))
random_std = stats.randint.rvs(low=1, high=5, size=K)  # Random uniformly distributed selections of signals
# Generate Standard Normally Distributed signals
for k in range(K):
    x[k] = np.random.normal(0, random_std[k], L)

y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
max_corr = Generate.generate_maxcorr(N, L, y)

G = Generate.generate_graph(max_corr, true_partition)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
pos = nx.spring_layout(G)
plt.title("Random Gaussian MRA samples")
nx.draw(G, pos, node_color=true_partition, node_size=20, edgelist=edges, edge_color=weights, width=1, cmap=plt.cm.jet, edge_cmap=plt.cm.Greens)
plt.show()