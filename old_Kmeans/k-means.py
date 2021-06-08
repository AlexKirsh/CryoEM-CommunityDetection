import igraph as ig
import networkx as nx
import numpy as np
import scipy as sp
import random

# steps:
# 1) pick k data points as initial centroids
# 2) Find the distance between each data point with the k-centroids
# 3) assign each data point to the closest centroid
# 4) Update centroid location by taking the average of the points in each cluster
# 5) repeat till no change or reaching to the max_iteration

#we use this function to 'delete' the None values in our simmilartiy function and convert it to the spectral dimension
#we convert the said matrix to a list of tupples, as we want to extract the labels efficiently
def modify_DataList(data_list,labels):
    n=len(data_list)
    #here we shall convert the matrix to the spectral dimension
    listy=[(labels[i],modified[i]) for i in range(n)]
    return listy


#We use this function to choose the k initial centers.
#k is the number of clusters
#l is the number of features
#moddified_data_list is a collection of n tuples: the first is the label of the point in the data,
#the second is the point's cooardinates (a 1Xn vector)
#initial_state may be:
#1. 'First': choose the k first vectors;
#2. 'Random' (Default): Choose uniformly k vectors from the n vectors;
def k_means_Type(initial_state,modified_data_list,k):
    initial_centers={}
    initial_centroids={}
    listy=[]
    n=len(modified_data_list)
    if (initial_state=='First'):
        listy=[(modified_data_list[i][0],modified_data_list[i][1]) for i in range(k)]
    else:
        Randomize=random.sample(range(n), k)
        Randomize.sort()
        listy=[(modified_data_list[val][0],modified_data_list[val][1]) for val in Randomize]

    for i in range(k):
        initial_centers[i]=listy[i][1]
        initial_centroids[i]=[listy[i][0]]
    return initial_centers, initial_centroids

#we use this function to define our metric of distance
#x and y are two vectors with dimension of 1Xn
#metric may be:
#Cosine: (x*y)/(||x||*||y||)
#Euclidean (default): d_i=sqrt(x_i^2-y_i^2)
def distance_metric(metric,x,y):
    if (metric=='Cosine'):
        x_norm=np.linalg.norm(x)
        y_norm=np.linalg.norm(y)
        scalar_product=np.dot(x, y)
        return scalar_product/(x_norm*y_norm)
    else:
        return np.linalg.norm(x-y)

#we use this function to compute the distances of the points from the centroids
#metric may be 'Cosine' or Euclidean (default)
#moddified_data_list is a collection of n tuples: the first is the label of the point in the data,
#the second is the point's cooardinates (a 1Xn vector)
#centers is a list of k points which we consider as centers
def Compute_Distances(metric,modified_data_list,centers,k):
    dist_dic={} #contains the distance of x[i] from all of the centers
    min_dict={} #contains [label in graph, the cluster with minimal distance, the minimal distance]
    n=len(modified_data_list)
    for i in range(n): #initiliazie the dictionaries
        dist_dic[i]=np.zeros(k)
        min_dict[i]=[modified_data_list[i],10**6,10**6] #[(label,cooardinates),closest centroid,min distance]
    for i in range(n):
        for j in range(k):
            dist_dic[i][j]=distance_metric(metric,modified_data_list[i][1],centers[j])
            if (dist_dic[i][j]<min_dict[i][2]): #min_dict[i][2] is the minimal distance (so far)
                min_dict[i][1]=j #save the label
                min_dict[i][2]=dist_dic[i][j] #save the distance
    return min_dict,dist_dic

#we use this function to refine our centroids and to calculate their cooardinates
#min_dict contains for each point label[i] in the data the minimal distance to a cluster
#and also the index with said distance
#k is the number of centers
def Label_centroids(min_dict,k):
    new_centroids={} #which vertex does the centroid contain
    new_centers={} #the cooardinates of the vertices inside the centers (which we will use soon)
    n=len(min_dict.keys())
    for j in range(k):
        new_centroids[j]=[]
        new_centers[j]=[]
    for i in range(n):
        new_centroids[min_dict[i][1]].append(min_dict[i][0][0]) #min_dict[i][1] is the label of the closest center, min_dict[i][0][0] is the label of the vertex
        new_centers[min_dict[i][1]].append(min_dict[i][0][1]) #min_dic[i][0][1] is the points cooardinates
    return new_centroids,new_centers

#calculates the new centers
#points_cooardinate are the cooardinates of the points that lie in the centroid.
#k is the number of centroids
def compute_new_centers(points_cooardinate,k):
    new_centers={}
    for i in range(k):
        new_centers[i]=np.mean(points_cooardinate[i],axis=0)
    return new_centers

#sub-methods:
def Clean_Empty_Clusters(cluster_dict,k):
    new_clusters={}
    counter=0
    for i in range(k):
        if(len(cluster_dict[i])!=0):
            new_clusters[counter]=cluster_dict[i]
            counter+=1
    return new_clusters

def begin_function(metric,initial_state):
    if (metric=="Cosine"):
        s1="Cosine"
    else:
        s1="Euclidean"
    if (initial_state=='First'):
        s2="For our initial values, we chose the first k vectors."
    else:
        s2 = "For our initial values, we chose k vectors uniformly."
    print("{0} Our distance metric is {1} distance.".format(s2,s1))

def Compare_Dicts(D1,D2):
    k1=len(D1.keys())
    k2 =len(D1.keys())
    if(k1!=k2):
        return False
    else:
        for i in range(k1):
            l1=len(D1[i])
            l2=len(D2[i])
            if(l1!=l2):
                return False
            else:
                List1=sorted(D1[i])
                List2=sorted(D2[i])
                if (List1!=List2):
                    return False
        return True

#need to standardize the data
#need to formula the data_list structure
def naive_k_means(k, data_list, max_iteration, metric,initial_state,labels=None,is_ok=True):
    print("Search for {0} clusters for at most {1} iterations".format(k,max_iteration))
    min_dict={}
    coordinates={}
    old_centroids={}
    cleaned_centroids={}
    n=len(data_list)
    if (labels is None):
        new_labels=[i for i in range(n)]
        print("As there where not initial labels, we assign to the vertex v_i the value i.")
    else:
        new_labels=labels
    if(k>=n):
        for i in range(n):
            old_centroids[i]=[new_labels[i]]
        print("The algorithm stopped after 0 iterations, as k>=n. Therefore there are at most {0} clusters".format(n))
        return old_centroids
    begin_function(metric,initial_state)
    modified_list = modify_DataList(data_list, new_labels)
    centers, old_centroids= k_means_Type(initial_state,modified_list,k) #(initial_centers, initial_centroids)
    for it in range(max_iteration):
        min_dict = Compute_Distances(metric,modified_list,centers,k)[0] #min_dict,dist_dic
        updated_centroids,coordinates = Label_centroids(min_dict,k) #new_centroids,new_centers
        cleaned_centroids=Clean_Empty_Clusters(updated_centroids,k)
        clean_len=len(cleaned_centroids.keys())
        if (clean_len<k):
            print("The algorithm stopped after {0} iterations and it suggests that there are in fact {1} clusters.".format(it,clean_len))
            return cleaned_centroids
        elif ((Compare_Dicts(cleaned_centroids,old_centroids) and (is_ok))):
            print("The algorithm stopped after {0} iterations as it arrived to saturation.".format(it))
            return cleaned_centroids
        else:
            old_centroids.clear()
            old_centroids=cleaned_centroids
            centers=compute_new_centers(coordinates,k)
    print("The algorithm stopped after reaching the maximal permitted number of iterations, which is {0}".format(max_iteration))
    return cleaned_centroids



