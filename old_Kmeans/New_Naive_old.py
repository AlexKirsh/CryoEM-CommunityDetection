import numpy as np
import random
import scipy.linalg as splin
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# steps:
# 1) pick k data points as initial centroids
# 2) Find the distance between each data point with the k-centroids
# 3) assign each data point to the closest centroid
# 4) Update centroid location by taking the average of the points in each cluster
# 5) repeat till no change or reaching to the max_iteration


#Auxiliary functions for simplicity
def tablelized(centers):
    m=len(centers)
    L=np.ndim(centers)
    new_centers=np.zeros((m,L+1))
    for i in range(m):
        for j in range(L):
            new_centers[i][j] = centers[i][1][j]
        new_centers[i][L]=centers[i][0]

    return new_centers


def Pack(addInitial, addDistance, addComunities, initial_centers, distances, communities):
    y=[]
    if addInitial:
        y.append(initial_centers)
    if addDistance:
        y.append(distances)
    if addComunities:
        communities_dicti=ToDicti(communities)
        y.append(communities_dicti)
    return tuple(y)

def ToDicti(communities):
    dicti={}
    indexy=0
    for i in range(len(communities)):
        indexy=communities[i][0]
        dicti[indexy]=communities[i][1]
    return dicti

#Sub methods
#We use this function to choose the k initial centers.
#initial_state may be:
#1. 'first': choose the k first vectors;
#2. 'random': Choose uniformly k vectors from the n vectors;
#3. 'mean': The centers are taken as k normal gaussian samples s.t N~(E[data_list],std(data_list))
#4. '++' (default): The initializion is according to the k-means++ algorithm
def k_means_Type(initial_state,data_list,k,n,metric,threshold):
    if initial_state=='first':
         return data_list[:k]
    elif initial_state=='random':
        Randomize=np.random.choice(range(n), k, replace=False)
        return [data_list[i] for i in Randomize]
    elif initial_state=='mean':
        mean = np.mean(data_list, axis=0)
        std = np.std(data_list, axis=0)
        centers = np.random.randn(k, np.shape(data_list)[1]) * std + mean
        return centers
    else:
        centers=[]
        random_index = np.random.choice(len(data_list),1)
        current_center=np.array(data_list[random_index][0])
        centers.append(current_center)
        counter=1
        #print(np.array(data_list[random_index][0]))
        #print(np.array(data_list[[random_index][0]]))
        while counter<k:
            dist_list=np.array([distance_metric(metric,data_list[i],current_center,threshold)
                                          for i in range(len(data_list))])
            sum_dist=sum(dist_list)
            probability_dist=dist_list/sum_dist
            chosen_index=np.random.choice(range(len(data_list)), 1, p=probability_dist) #np.random.choice receives
            # only 1-D array
            #current_center = np.array(data_list[random_index][0])
            current_center=data_list[chosen_index[0]]
            centers.append(current_center)
            counter+=1
        return centers



#we use this function to define our metric of distance
#x and y are two vectors with dimension of 1Xn
#metric may be:
#cosine:d_i^2=1-(x*y)/(||x||*||y||)
#manhattan: d_i=|x_i-y_i|
#pearson: d_i^2= 1-((x-E[x])*(y-E[y]))/(||x-E[x]||*||y-E[y]||)
#circular_ciorr: d_i^2 = max(ifft(fft(y[i]).conj() * fft(y[j])).real)
#euclidean (default): d_i=sqrt(x_i^2-y_i^2)
def distance_metric(metric,x,y,threshold=0):
    if metric == 'cosine':
        x_norm=np.linalg.norm(x)
        y_norm=np.linalg.norm(y)
        scalar_product=np.dot(x, y)
        return 1-scalar_product/(x_norm*y_norm)
    elif metric == 'manhattan':
        return np.linalg.norm(x - y, ord=1)
    elif metric == 'pearson':
        Ex=np.mean(x)
        Ey=np.mean(y)
        x_norm=np.linalg.norm(x-Ex)
        y_norm=np.linalg.norm(y-Ex)
        scalar_product = np.dot(x-Ex, y-Ex)
        return 1-scalar_product/(x_norm*y_norm)
    elif metric == 'circular':
        circular=max(ifft(fft(x).conj() * fft(y).real).real)
        if circular<threshold:
            return 2*len(x)
        return 2*len(x)*(1-circular)
    else:
        return np.linalg.norm(x-y)

#assign each point to its closest centroid
def assign_centroids(centers,data_list,metric,threshold):
    centroids={}
    demi_centroids={}
    n = len(data_list)
    m= len(centers)
    index=0
    labeled=[]
    distances=[]
    for i in range(m):
        centroids[i]=[centers[i],[]]
        demi_centroids[i] = []
    for i in range(n):
        dist_list=np.array([distance_metric(metric,data_list[i],centers[j],threshold) for j in range(m)],dtype=object)
        index=int(np.argmin(dist_list))
        labeled.append(index)
        distances.append(dist_list[index])
        centroids[index][1].append(data_list[i])
        demi_centroids[index].append((data_list[i],i))
    return centroids, labeled, distances, demi_centroids  #centroids: (index: [center,points])

#change the affinity of the point in the label list and in the distance list
def change_label_and_distances(point,originial_index, new_index,labeled,distances,demi_centroids):
    originial_community=demi_centroids[originial_index]
    for i in range(len(originial_community)):
        if np.array_equal(originial_community[i][0],point):
            labeled[originial_community[i][1]]=new_index
            distances[originial_community[i][1]]=0
            break

#we use this function to treat empty cases:
#'none': ignore them (an exception might be thrown)
#'add_farthest': take the farthest point from its corresponding cluster and add it to an empty cluster
#'add_random': take a random point from a big cluster and move it to the empty cluster
#'remove': delete the empty clusters and reduce the number of cluster
#'add_biggest' (default): take the farthest point from the largest cluster and add it to an empty cluster
def treat_empty(centroids, labeled, distances, demi_centroids, metric,threshold, method):
    centers, communities = flatten_centroid1(centroids)
    max_points=[]
    maximal_distance_point=0
    biggest=0
    biggest_index=0
    k=len(centers)
    new_centroids={}
    options=[]
    counter=0
    if (method=='none'):
        return centroids, labeled, distances

    if method=='remove':
        for i in range(k):
            if len(communities[i][1])>0 :
                new_centroids[counter]=[centers[i][1], communities[i][1]]
        return new_centroids, labeled, distances
    # else: method == 'add_random', 'add_biggest' or 'add_farthest'
    for i in range(k):
        if len(communities[i][1])==0:
            if method=='add_farthest' or method=='add_biggest':
                if method == 'add_farthest':
                    for j in range(k):
                        if len(communities[j][1])>1:
                            dist_list=np.array([distance_metric(metric,point,centers[j][1],threshold)
                                                for point in communities[j][1]],dtype=object)
                            max_index=np.argmax(dist_list)
                            max_points.append((communities[j][1][max_index],max_index,dist_list[max_index],
                                               communities[j][0]))
                            # point, the index of the max distance, max distance, index of community
                    maximal_distance_point = max(max_points, key=lambda x: x[2])
                    max_points.clear()
                elif method== 'add_biggest':
                    for j in range(k):
                        if len(communities[j][1])>biggest:
                         biggest=len(communities[j][1])
                         biggest_index=j
                    dist_list = np.array([distance_metric(metric, point, centers[biggest_index][1], threshold)
                                          for point in communities[biggest_index][1]],dtype=object)
                    max_index = np.argmax(dist_list)
                    maximal_distance_point=(communities[biggest_index][1][max_index], max_index, dist_list[max_index],
                                            communities[biggest_index][0])
                    # point, the index of the max distance, max distance, index of community
                communities[i][1].append(maximal_distance_point[0])
                centers[i][1]=maximal_distance_point[0]
                communities[maximal_distance_point[3]][1].pop(int(maximal_distance_point[1]))
                change_label_and_distances(maximal_distance_point[0],maximal_distance_point[3],communities[i][0],
                                           labeled,distances,demi_centroids)
            else:
                if method=='add_random':
                    for j in range(k):
                        if len(communities[j][1])>1:
                            options.append(j)
                    random_community_index = np.random.choice(options,replace=False)
                    random_point_index = np.random.sample(range(len(communities[random_community_index][1])),1)[0]
                    communities[i][1].append(communities[random_community_index][1][random_point_index])
                    centers[i][1]=communities[random_community_index][1][random_point_index]
                    communities[random_community_index][1].pop(int(random_point_index))
                    change_label_and_distances(communities[i][1],random_community_index,communities[i][0],labeled,
                                               distances,demi_centroids)

    for i in range(k):
        new_centroids[i]=[centers[i][1],communities[i][1]]

    return new_centroids,labeled, distances

#Assign points to centroids, while assuring now empty cluster is created
def Designate_Centroids(centers,data_list,metric,threshold,method):
    new_centroids, labeled, distances, demi_cetnroids = assign_centroids(centers,data_list,metric,threshold)
    new_centroids, labeled, distances = treat_empty(new_centroids,labeled,distances,demi_cetnroids,metric,threshold,
                                                    method)
    return new_centroids, labeled, distances

#update the centers of the centroids
def compute_new_centers(centroids,metric,threshold):
    new_centroids={}
    for key,value in centroids.items():
        if(len(value[1])==0):
            new_center=value[0]
        else:
            new_center=np.mean(value[1],axis=0)
        new_centroids[key]=[new_center,value[1]]
    return new_centroids #(index: [center,points])

#check if we arrived to saturation
def compare_centroids(old_centroids,new_centroids):
    centers1=flatten_centroid2(old_centroids)
    centers2 = flatten_centroid2(new_centroids)
    t1=all([np.allclose(x, y) for x, y in zip(centers1, centers2)])
    return t1 and len(centers1) > 0 and len(centers2) > 0

#For output
def begin_function(metric,initial_state,k,max_iteration, n, method, threshold, addInitial, addDistance, addComunities):

    print("Search for {0} clusters for at most {1} iterations:".format(k, max_iteration))
    if n==k:
        print("As the number of requested communities is equal to the number of the data points, "
              "the solution is trival.")
    else:
        if metric=="cosine":
            s1="the Cosine distance"
        elif metric=='manhattan':
            s1='the Manhattan distance'
        elif metric == 'pearson':
            s1='the Pearson-correlated distance'
            if threshold != 0:
                s1 += ' with threshold value of {0}'.format(threshold)
        elif metric == 'circular':
            s1='the circular-correlated distance'
            if threshold != 0:
                s1 += ' with threshold value of {0}'.format(threshold)
        else:
            s1='the Euclidean distance'

        if initial_state=='first':
            print("*For our initial values, we chose the first k vectors.")
        elif initial_state == 'random':
            print("*For our initial values, we chose k vectors uniformly.")
        elif initial_state == 'mean':
            print("*For our initial values, we chose k samples from a normal distribution which reflects the data")
        else:
            print("Our initialization of centers follows the k-means++ algorithm")

        print("*Our distance metric is the {0}".format(s1))

        if method=='none':
            print("*We will allow empty clusters")
        elif method=='add_farthest':
            print("*We will asign the farthest points from the clusters' centers to empty clusters")
        elif method=='random':
            print("*Random points will be assigned to empty clusters")
        elif method == 'remove':
            print("*Empty clusters will be deleted")
        else:
            print("*We will assign the farthest points from the biggest cluster to empty clusters")

        if addComunities:
            print("*A dictionary containing the partition to the communities will be added to the output")
        if addInitial:
            print("*The initial centers that were chosen will be added to the output")
        if addDistance:
            print("*The distances of each point from the final centers will be added to the output")

#retrieve the centers and the communities separated
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

#core:
#input:
#-------
#data_list: case 1: NXL
#k: number of clusters
#max_iteration: an upper bound for the number of iterations the algorithm will run. if not specified,
# the algorithm should run until saturation (no change occurs in the centers cooardinates)
#initial_state: should the initial choice of the centers follow the rule:
# 1) 'first': should we take the k-first vectors in the data
# 2) 'random': should we take k ranodom vectors (drown uniformly) from the data
# 3) 'mean': should we reflect the k-centers as k-samples from a normal distribution N~(E[vectors],std[vectors])?
# 4) '++' (default): should we follow the k-means++ initialization?
# metric: which distance metric should be applied: the choices are:
#1) The Cosine distance
#2) The Manhattan distance (L1 norm)
#3) The pearson correlation distance
#4) The circular-correlation distance
#5) The Euclidean Distance (default)
# threshold (optional) if we choice correlation distance, from which value we will consider
# the distance to be valid (default: 0)
# method: define what to do if an empty cluster emerges:
#1) 'none': ignore them (an exception might be thrown)
#2) 'add_farthest': take the farthest point from its corresponding cluster and add it to an empty cluster
#3)'add_random': take a random point from a big cluster and move it to the empty cluster
#4) 'remove': delete the empty clusters and reduce the number of cluster
#5) 'add_biggest' (default): take the farthest point from the largest cluster and add it to an empty cluster
# addInitial (Optional): if we want to add the initial choice of the centers to the output (default: no)
# addDistance (Optional): if we want to add the final distance of each point from its centroid (default: no)
# addComunitie (Optional): if we want to add the output a different ordering of the points, in which we assign
# each centroid its index and all of the points which are in its interior
#in a dictionary form
#output:
#-------
# centers, labeled, optional[0], optional[1], optional[2] (see the explanation in the input section)
# A list of the centers of the final centroids
# A list of labels which indicates for each signal which is its closest centroid (the order is preserved as
# it was in the input)

def Naive_k_means(data_list,k,max_iteration=300,initial_state='++',metric='euclidean',threshold=0, method='none',
                  addInitial=False, addDistance=False, addCommunities=False):
    old_centroids={}
    new_centroids={}
    centers=[]
    communities=[]
    labeled = []
    distances=[]
    counter = 0
    n = len(data_list)
    features_counter=[addInitial,addCommunities,addDistance].count(True)

    if n<k:
        print("The number of communities cannot be bigger than the dimension of the data.")
        if features_counter == 0:
            return None, None
        elif features_counter == 1:
            return None, None, None
        elif features_counter == 2:
            return None, None, None, None
        else:
            return None, None, None, None, None

    begin_function(metric,initial_state,k,max_iteration,n, method,threshold,addInitial,addDistance,addCommunities)
    if n == k:
        for i in range(n):
            centers.append(data_list[i])
            communities.append(data_list[i])
            labeled.append(i)
            distances.append(0)
        *x, = Pack(addInitial,addDistance,addCommunities,centers,distances,communities)
        if features_counter == 0:
            return centers, labeled
        elif features_counter == 1:
            return centers, labeled, x[0]
        elif features_counter == 2:
            return centers, labeled, x[0], x[1]
        else:
            return centers, labeled, x[0], x[1], x[2]

    centers=k_means_Type(initial_state,data_list,k,n,metric,threshold)
    initial=[(i,centers[i]) for i in range(k)]
    initial=tablelized(initial)
    new_centroids, labeled, distances, _ =assign_centroids(centers,data_list,metric,threshold)
    while True:
        centers, communities = flatten_centroid1(old_centroids)
        if compare_centroids(old_centroids, new_centroids):
            print("The algorithm stopped after {0} iterations as it arrived to saturation.".format(counter))
            centers=tablelized(centers)
            *x, = Pack(addInitial, addDistance, addCommunities, initial, distances, communities)
            if features_counter == 0:
                return centers, labeled
            elif features_counter == 1:
                return centers, labeled, x[0]
            elif features_counter == 2:
                return centers, labeled, x[0], x[1]
            else:
                return centers, labeled, x[0], x[1], x[2]
        else:
            old_centroids=new_centroids
            new_centroids=compute_new_centers(old_centroids,metric,threshold)
            centers=flatten_centroid2(new_centroids)
            new_centroids, labeled, distances=Designate_Centroids(centers,data_list,metric,threshold,method)
            counter+=1
            if (counter==max_iteration):
                break
    print("The algorithm stopped after reaching the maximal permitted number of iterations, which is {0}".format(
        counter))
    centers, communities = flatten_centroid1(old_centroids)
    centers=tablelized(centers)
    *x, = Pack(addInitial, addDistance, addCommunities, initial, distances, communities)
    if features_counter == 0:
        return centers, labeled
    elif features_counter == 1:
        return centers, labeled, x[0]
    elif features_counter == 2:
        return centers, labeled, x[0], x[1]
    else:
        return centers, labeled, x[0], x[1], x[2]
