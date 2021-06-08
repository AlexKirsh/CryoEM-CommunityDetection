import numpy as np
import random
import scipy.linalg as splin
from numpy.fft import fft, ifft


# --------------------------------------------Distance Computation------------------------------------------------------
def cosine_distance(x, y, threshold):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    scalar_product = np.dot(x, y)
    pre_dist = scalar_product / (x_norm * y_norm)
    if pre_dist < threshold:
        return 1
    return 1 - pre_dist


def pearson_distance(x, y, threshold):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    scalar_product = np.dot(x, y)
    pre_dist = scalar_product / (x_norm * y_norm)
    if pre_dist < threshold:
        return 1
    return 1 - pre_dist


def manhattan_distance(x, y, threshold):
    pre_dist = np.linalg.norm(x - y, ord=1)
    if pre_dist < threshold:
        return np.infty
    return pre_dist


def euclidean_distance(x, y, threshold):
    corr_arg = np.argmax(ifft(fft(x).conj() * fft(y)).real)
    pre_dist = np.linalg.norm(np.roll(x, int(corr_arg)) - y)
    if pre_dist < threshold:
        return np.infty
    return pre_dist


def compute_distance(point, center, feature_length, threshold, metric, circular):
    if circular == 'on':
        if metric == 'cosine':
            d_list = [cosine_distance(np.roll(point, shift), center, threshold) for shift in range(feature_length)]
        elif metric == 'pearson':
            d_list = [pearson_distance(np.roll(point, shift), center, threshold) for shift in range(feature_length)]
        elif metric == 'manhattan':
            d_list = [manhattan_distance(np.roll(point, shift), center, threshold) for shift in range(feature_length)]
        else:
            #d_list = [euclidean_distance(np.roll(point, shift), center, threshold) for shift in range(feature_length)]
            d_list = euclidean_distance(point, center, threshold)
        #return np.min(d_list)
        return d_list
    else:
        if metric == 'cosine':
            return cosine_distance(point, center, threshold)
        elif metric == 'pearson':
            return pearson_distance(point, center, threshold)
        elif metric == 'manhattan':
            return manhattan_distance(point, center, threshold)
        else:
            return euclidean_distance(point, center, threshold)


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------Initialization----------------------------------------------------------
def centers_initialization(data_list, k, way, feature_length, threshold, metric, circular):
    n = np.shape(data_list)[0]

    if way == 'first':
        return np.asarray(data_list[:k], dtype=np.ndarray)

    elif way == 'random':
        randomize = np.random.choice(range(n), k, replace=False)
        return np.asarray([data_list[i] for i in randomize], dtype=np.ndarray)

    elif way == 'mean':
        mean = np.mean(data_list, axis=0)
        std = np.std(data_list, axis=0)
        centers = np.random.randn(k, feature_length) * std + mean
        return np.asarray(centers, dtype=np.ndarray)

    else:
        centers = []
        random_index = np.random.choice(range(n))
        current_center = np.array(data_list[random_index])
        counter = 0
        while counter < k:
            dist_list = np.array(
                [compute_distance(data_list[i], current_center, feature_length, threshold, metric, circular)
                 for i in range(n)])
            sum_dist = sum(dist_list)
            probability_dist = dist_list / sum_dist
            chosen_index = np.random.choice(n, 1, p=probability_dist)  # np.random.choice receives
            # only 1-D array
            current_center = data_list[chosen_index[0]]
            centers.append(current_center)
            counter += 1
        return np.asarray(centers, dtype=np.ndarray)


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------Check Saturation--------------------------------------------------------
def isSaturated(centers_1, centers_2, way):
    c1 = len(centers_1)
    c2 = len(centers_2)
    if (c1 != c2):
        return False
    if way == 'exact':
        for i in range(c1):
            current = centers_1[i]
            close_centers_self = [1 for j in range(c1) if np.array_equiv(current, centers_1[j])]
            close_centers_other = [1 for j in range(c2) if np.array_equiv(current, centers_2[j])]
            if len(close_centers_other) != len(close_centers_self):
                return False
        return True
    else:
        for i in range(c1):
            current = np.asarray(centers_1[i], dtype=float)
            close_centers_self = [1 for j in range(c1) if np.isclose(current,
                                                                     np.asarray(centers_1[j],
                                                                                dtype=float)).all()]
            close_centers_other = [1 for j in range(c2) if np.isclose(current,
                                                                      np.asarray(centers_2[j],
                                                                                 dtype=float)).all()]
            if len(close_centers_other) != len(close_centers_self):
                return False
        return True


# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------Assign Points to Centroids------------------------------------------------------
def assign(data_list, current_centers, features_length, threshold, metric, circular):
    data_length = len(data_list)
    centers_length = len(current_centers)
    new_centroids = {}
    new_labels = np.zeros(data_length)
    # initialize dictionary
    for i in range(centers_length):
        new_centroids[i] = []

    for i in range(data_length):
        distances_from_centers = np.asarray(
            [compute_distance(data_list[i], current_centers[j], features_length, threshold,
                              metric, circular) for j in range(centers_length)])
        minimal_distance_index = int(np.argmin(distances_from_centers))
        new_labels[i] = minimal_distance_index
        new_centroids[minimal_distance_index].append(
            i)  # each centroid contains the indices of the points in the data list

    for i in range(centers_length):
        new_centroids[i] = np.array(new_centroids[i])

    return new_labels, new_centroids


# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------Handle empty cases--------------------------------------------------------------
def remove_empty(centroids, centers, data_length, centers_length):
    modified_centroids = {}
    modified_labels = np.zeros(data_length)
    modified_centers = []
    counter = 0
    for i in range(len(centroids.keys())):
        inner_points_no = len(centroids[i])
        if inner_points_no > 0:
            modified_centroids[counter] = centroids[i]
            modified_centers.append(centers[i])
            for point_index in centroids[i]:
                modified_labels[point_index] = counter
            counter += 1
    modified_centers = np.asarray(modified_centers, dtype=np.ndarray)
    return modified_labels, modified_centroids, modified_centers


def move_farthest_to_empty(centroids, centers, deux_list, labels, data_length, features_length, centers_length,
                           threshold, metric, circular):
    oddest_point_indices = []  # (centroid_index,internal_index)
    oddest_point_distance_list = []
    maximal_index = 0
    oddest_point_distance_list = []
    oddest_point_indices = []
    farthest_distance_index = 0
    oddest_point_internal_index = 0
    oddest_point_external_index = 0
    oddest_center = 0

    for i in range(centers_length):
        if len(centroids[i]) == 0:
            for j in range(centers_length):
                if len(centroids[j]) > 1:
                    distances_from_respective_center = [
                        compute_distance(deux_list[index][0], centers[j], features_length,
                                         threshold, metric, circular)
                        for index in centroids[j]]
                    maximal_index = int(np.argmax(distances_from_respective_center))
                    oddest_point_distance_list.append(distances_from_respective_center[maximal_index])
                    oddest_point_indices.append((j, maximal_index))
            farthest_distance_index = int(np.argmax(oddest_point_distance_list))
            oddest_point_internal_index = oddest_point_indices[farthest_distance_index][1]
            oddest_center = oddest_point_indices[farthest_distance_index][0]
            oddest_point_external_index = centroids[oddest_center][oddest_point_internal_index]
            dup = centroids[i].tolist()
            dup.append(oddest_point_external_index)
            centroids[i] = np.asarray(dup)
            centers[i] = deux_list[oddest_point_external_index][0]
            labels[oddest_point_external_index] = i
            centroids[oddest_center] = np.delete(centroids[oddest_center], oddest_point_internal_index)
        maximal_index = 0
        oddest_point_distance_list.clear()
        oddest_point_indices.clear()
        farthest_distance_index = 0
        oddest_point_internal_index = 0
        oddest_point_external_index = 0
        oddest_center = 0

    return labels, centroids, centers


def move_biggest_to_empty(centroids, centers, deux_list, labels, data_length, features_length, centers_length,
                          threshold, metric, circular):
    biggest_index = 0
    for i in range(centers_length):
        if len(centroids[i]) == 0:
            size_list = [len(centroids[j]) for j in range(centers_length)]
            biggest_index = int(np.argmax(size_list))
            distances_from_respective_center = [compute_distance(deux_list[index][0], centers[biggest_index],
                                                                 features_length, threshold, metric, circular)
                                                for index in centroids[biggest_index]]
            farthest_internal_index = int(np.argmax(distances_from_respective_center))
            farthest_external_index = centroids[biggest_index][farthest_internal_index]
            dup = centroids[i].tolist()
            dup.append(farthest_external_index)
            centroids[i] = np.asarray(dup)
            centers[i] = deux_list[farthest_external_index][0]
            labels[farthest_external_index] = i
            centroids[biggest_index] = np.delete(centroids[biggest_index], farthest_internal_index)
    return labels, centroids, centers


def split_to_empty(centroids, centers, deux_list, labels, data_length, features_length, centers_length,
                   threshold, metric, circular):
    Empty_indices = []
    biggest_index = 0
    biggest_len = 0
    for i in range(centers_length):
        if len(centroids[i]) == 0:
            Empty_indices.append(i)
        else:
            if len(centroids[i]) > biggest_len:
                biggest_len = len(centroids[i])
                biggest_index = i
    empty_clusters = len(Empty_indices)
    points_to_contirbute = biggest_len // (empty_clusters + 1)
    farthest_points_distances = [
        [i, compute_distance(centroids[biggest_index][i], centers[biggest_index], features_length,
                             threshold, metric, circular)] for i in range(biggest_len)]
    sorted(farthest_points_distances, key=lambda x: x[1])
    farthest_points_distances.reverse()
    farthest_points_distances=farthest_points_distances[:(points_to_contirbute*empty_clusters)]
    np.random.shuffle(farthest_points_distances)
    orig_len=len(farthest_points_distances)
    farthest_points_distances_new=[farthest_points_distances[i][0] for i in range(orig_len)]
    points_to_disperse=list(centroids[biggest_index])
    for i in range(empty_clusters):
        dup = centroids[Empty_indices[i]].tolist()
        for j in range(points_to_contirbute):
            popped = farthest_points_distances_new.pop()
            external_index=centroids[biggest_index][popped]
            labels[external_index] = Empty_indices[i]
            points_to_disperse.remove(external_index)
            dup.append(popped)
        centroids[i] = np.array(sorted(dup), dtype=int)
    centroids[biggest_index] = np.array(sorted(points_to_disperse),dtype=int)
    centers = update_centers(centroids, centers, deux_list)

    return labels, centroids, centers


def move_random_to_empty(centroids, centers, deux_list, labels, data_length, features_length, centers_length,
                         threshold, metric, circular):
    for i in range(centers_length):
        if len(centroids[i]) == 0:
            options = [j for j in range(centers_length) if len(centroids[j]) > 1]
            random_community_index = np.random.choice(options)
            random_internal_point_index = np.random.choice(range(len(centroids[random_community_index])))
            random_external_point_index = centroids[random_community_index][random_internal_point_index]
            labels[random_external_point_index] = i
            dup = centroids[i].tolist()
            dup.append(random_external_point_index)
            centroids[i] = np.asarray(dup)
            centers[i] = deux_list[random_external_point_index][0]
            centroids[random_community_index] = np.delete(centroids[random_community_index],
                                                          random_internal_point_index)
    return labels, centroids, centers


def handle_empty(centroids, centers, deux_list, labels, data_length, features_length, centers_length,
                 threshold, metric, circular, empty_procedure):
    if empty_procedure == 'remove':
        return remove_empty(centroids, centers, data_length, centers_length)
    elif empty_procedure == 'biggest':
        return move_biggest_to_empty(centroids, centers, deux_list, labels, data_length, features_length,
                                     centers_length,
                                     threshold, metric, circular)
    elif empty_procedure == 'farthest':
        return move_farthest_to_empty(centroids, centers, deux_list, labels, data_length, features_length,
                                      centers_length,
                                      threshold, metric, circular)
    elif empty_procedure == 'random':
        return move_random_to_empty(centroids, centers, deux_list, labels, data_length, features_length,
                                    centers_length,
                                    threshold, metric, circular)
    elif empty_procedure == 'split':
        return split_to_empty(centroids, centers, deux_list, labels, data_length, features_length,
                              centers_length,
                              threshold, metric, circular)
    else:  # ignore
        return labels, centroids, centers


def advance_assign(data_list, current_centers, deux_list, features_length, threshold, metric, circular,
                   empty_procedure):
    data_length = np.shape(data_list)[0]
    centers_length = np.shape(current_centers)[0]
    new_labels, new_centroids = assign(data_list, current_centers, features_length, threshold, metric, circular)
    return handle_empty(new_centroids, current_centers, deux_list, new_labels, data_length, features_length,
                        centers_length,
                        threshold, metric, circular, empty_procedure)


# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------Compute New Centers-------------------------------------------------------------
def update_centers(centroids, old_centers, deux_list):
    centers_length = len(centroids.keys())
    new_centers = np.zeros(centers_length, dtype=np.ndarray)
    for i in range(centers_length):
        if len(centroids[i]) == 0:
            new_centers[i] = old_centers[i]
        else:
            new_centers[i] = np.mean([deux_list[index][0] for index in centroids[i]], axis=0)
    return new_centers


# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------Extra---------------------------------------------------------------
def modified_output(centers, centroids, labels, initial_centers, as_dict, with_initial):
    if not with_initial:
        if not as_dict:
            return labels, centers
        else:
            return centroids, centers
    else:
        if not as_dict:
            return labels, centers, initial_centers
        else:
            return centroids, centers, initial_centers


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------Main--------------------------------------------------------------
# 1) pick k data points as initial centroids
# 2) Find the distance between each data point with the k-centroids
# 3) assign each data point to the closest centroid
# 4) Update centroid location by taking the average of the points in each cluster
# 5) repeat till no change or reaching to the max_iteration

def final_kmeans(datalist, k, max_iteration=300, initialization_type='++', metric='euclidean', threshold=0,
                 circular='on', empty_procedure='ignore', saturation='approx',
                 plot_initial='off', plot_asdictionary=False, report_process=False):
    n = np.shape(datalist)[0]
    L = np.shape(datalist)[1]
    labels = np.zeros(n)
    old_centers = np.zeros((k,), dtype=np.ndarray)
    centroids = {}
    initial_centers = np.zeros((k,), dtype=np.ndarray)
    new_centers = np.zeros((k,), dtype=np.ndarray)
    duex_list = [(datalist[i], i) for i in range(n)]

    # ---------------------initialize the first k-centers------------------------
    initial_centers = centers_initialization(datalist, k, initialization_type, L, threshold, metric, circular)
    new_centers = np.copy(initial_centers)

    # -------------------------First Iteration Update----------------------------
    labels, centroids, new_centers = advance_assign(datalist, new_centers, duex_list, L,
                                                    threshold, metric, circular, empty_procedure)
    # -------------------------Loop untill saturation----------------------------
    for iter in range(max_iteration):
        if iter > 0:
            if isSaturated(old_centers, new_centers, way=saturation):
                print("The algorithm stopped after {0} iterations due to saturation".format(iter))
                return modified_output(new_centers, centroids, labels, initial_centers, plot_asdictionary, plot_initial)
        old_centers = np.copy(new_centers)

        labels, centroids, new_centers = advance_assign(datalist, old_centers, duex_list, L,
                                                        threshold, metric, circular, empty_procedure)
        new_centers = update_centers(centroids, new_centers, duex_list)

    # -------------------------Finished without saturation-------------------------
    print("The algorithm stopped after it arrives to the maximal permitted saturation, which is {0}"
          .format(max_iteration))
    return modified_output(new_centers, centroids, labels, initial_centers, plot_asdictionary, plot_initial)
# ----------------------------------------------------------------------------------------------------------------------
