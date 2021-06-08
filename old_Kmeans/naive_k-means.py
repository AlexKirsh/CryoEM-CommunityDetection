import igraph as ig
import networkx as nx
import numpy as np
import random
import scipy.linalg as splin
import matplotlib.pyplot as plt

# steps:
# 1) pick k data points as initial centroids
# 2) Find the distance between each data point with the k-centroids
# 3) assign each data point to the closest centroid
# 4) Update centroid location by taking the average of the points in each cluster
# 5) repeat till no change or reaching to the max_iteration


#We use this function to choose the k initial centers.
#initial_state may be:
#1. 'first': choose the k first vectors;
#2. 'random' (Default): Choose uniformly k vectors from the n vectors;
def k_means_Type(initial_state,data_list,k,n):
    if (initial_state=='first'):
         return data_list[:k]
    else:
        Randomize=random.sample(range(n), k)
        return [data_list[i] for i in Randomize]

#we use this function to define our metric of distance
#x and y are two vectors with dimension of 1Xn
#metric may be:
#cosine: 1-(x*y)/(||x||*||y||)
#manhattan: d_i=|x_i-y_i|
#euclidean (default): d_i=sqrt(x_i^2-y_i^2)
def distance_metric(metric,x,y):
    if (metric == 'cosine'):
        x_norm=np.linalg.norm(x)
        y_norm=np.linalg.norm(y)
        scalar_product=np.dot(x, y)
        return (1-scalar_product/(x_norm*y_norm))
    elif (metric == 'manhattan'):
        return(np.linalg.norm(x-y, ord=1))
    else:
        return np.linalg.norm(x-y)

#assign each point to its closest centroid
def assign_centroids(centers,data_list,metric,n):
    centroids={}
    m= len(centers)
    index=0
    for i in range(m):
        centroids[i]=[centers[i],[]]
    for i in range(n):
        dist_list=[distance_metric(metric,data_list[i],centers[j]) for j in range(len(centers))]
        index=np.argmin(dist_list)
        centroids[index][1].append(data_list[i])
    return centroids  #(index: [center,points])

#update the centers of the centroids
def compute_new_centers(centroids):
    new_centroids={}
    for key,value in centroids.items():
        new_center=np.mean(value[1],axis=0)
        new_centroids[key]=[new_center,value[1]]
    return new_centroids #(index: [center,points])

#check if we arrived to saturation
def compare_centroids(old_centroids,new_centroids):
    centers1=flatten_centroid2(old_centroids)
    centers2 = flatten_centroid2(new_centroids)
    t1=all([np.allclose(x.sort(), y.sort()) for x, y in zip(centers1, centers2)])
    return (t1 and len(centers1)>0 and len(centers2)>0)

#For output
def begin_function(metric,initial_state,k,max_iteration,n):
    if(max_iteration!=None):
        print("Search for {0} clusters for at most {1} iterations:".format(k, max_iteration))
    else:
        print("Search for {0} clusters:".format(k))
    if(n==k):
        print("As the number of requested communities is equal to the number of the data points, the solution is trival.")
    else:
        if (metric=="cosine"):
            s1="cosine"
        elif (metric=='manhattan'):
            s1='manhattan'
        else:
            s1="euclidean"
        if (initial_state=='first'):
            print("*For our initial values, we chose the first k vectors.")
        else:
            print("*For our initial values, we chose k vectors uniformly.")
        print("*Our distance metric is the '{0}' distance.".format(s1))

#retrieve the centers and the communities seperated
def flatten_centroid1(centroid):
    centers=[]
    communities=[]
    for key,value in centroid.items():
        centers.append([key,value[0]])
        communities.append([key,value[1]])
    return centers, communities

#retrieve the centers without labels
def flatten_centroid2(centroid):
    return [centroid[key][0] for key in centroid.keys()]

def Naive_k_means(data_list,k,max_iteration=None,initial_state='random',metric='euclidean'):
    old_centroids={}
    new_centroids={}
    centers=[]
    communities=[]
    n = len(data_list)
    counter=0
    if(n<k):
        print("The number of communities cannot be bigger than the dimension of the data.")
        return None, None
    begin_function(metric,initial_state,k,max_iteration,n)
    if(n==k):
        for i in range(n):
            centers.append(data_list[i])
            communities.append(data_list[i])
        return centers,communities
    centers=k_means_Type(initial_state,data_list,k,n)
    new_centroids=assign_centroids(centers,data_list,metric,n)
    while(True):
        centers, communities = flatten_centroid1(old_centroids)
        if(compare_centroids(old_centroids,new_centroids)):
            print("The algorithm stopped after {0} iterations as it arrived to saturation.".format(counter))
            return centers, communities
        else:
            old_centroids=new_centroids
            new_centroids=compute_new_centers(old_centroids)
            centers=flatten_centroid2(new_centroids)
            new_centroids=assign_centroids(centers,data_list,metric,n)
            counter+=1
            if (max_iteration is not None):
                if(counter==max_iteration):
                    break
    print("The algorithm stopped after reaching the maximal permitted number of iterations, which is {0}".format(
        counter))
    return(flatten_centroid1(old_centroids))

#-------for spectral analysis if needed--------
# Find D_inverse and id matrix for computing the normalized laplacian
def Build_auxilary_matrices(data_list,n):
    modified=np.array(data_list)
    D_inverse = np.zeros((n, n))
    Identity = np.zeros((n, n))
    counter=0
    for i in range(n):
        counter=len(modified[i].nonzero()[0])
        if(counter>0):
            D_inverse[i][i]=1/counter
        else:
            D_inverse[i][i]=0
        Identity[i][i]=1
    return D_inverse,Identity


def Row_normalization(M):
    row_sums = M.sum(axis=1)
    new_matrix = M / row_sums[:, np.newaxis]
    return new_matrix

# we use this function to create M_(nXk) matrix of the k eigen vectors
#This matrix is the input to the k-means algorithm
def Compute_Normalize_Laplacian(k,data_list,n):
    D_inverse, I = Build_auxilary_matrices(data_list,n)
    L=I-(D_inverse @ data_list)
    eigenvalue, eigenvecs=splin.eigh(L,eigvals=(n-k,n-1)) #eigenvectors:
    return (Row_normalization(eigenvecs))

