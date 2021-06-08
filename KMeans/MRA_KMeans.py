import Final_Kmeans
import MRA_Samples
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import New_Naiv_non_empty

def plot_nmi_scores(N, sigma_list, runs_num, MRA_type):
    nmi_scores = []
    snr_list = []
    for sigma in sigma_list:
        nmi_samples_1 = []
        nmi_samples_2 = []
        nmi_samples_3 = []
        print("sigma: {}".format(sigma))

        if MRA_type == "Rect_Trian":
            K = 2
            y, corr, true_labels = MRA_Samples.MRA_Rect_Trian(N, L=50, K=K, sigma=sigma)
        elif MRA_type == "Standard Normal":
            K = 10
            y, corr, true_labels = MRA_Samples.MRA_StandardNormal(N, L=50, K=K, sigma=sigma)

        print("true: {}".format(true_labels))

        kmeans_partitions = []
        kmeans_partitions_pp = []
        kmeans_partitions_empty = []

        # Calculate NMI score for a number (samples_num) of random graphs
        for i in range(runs_num):
            print("iteration number: {}".format(i))

            kmeans_partition = Final_Kmeans.final_kmeans(y, K, max_iteration=100, initialization_type="random", empty_procedure='ignore', circular='on')
            kmeans_partitions.append(kmeans_partition[0])
            nmi_samples_1.append(normalized_mutual_info_score(true_labels, kmeans_partition[0]))
            kmeans_partition = Final_Kmeans.final_kmeans(y, K, max_iteration=100, empty_procedure='ignore', circular='on')
            kmeans_partitions_pp.append(kmeans_partition[0])
            nmi_samples_1.append(normalized_mutual_info_score(true_labels, kmeans_partition[0]))
            kmeans_partition = Final_Kmeans.final_kmeans(y, K, max_iteration=100, empty_procedure='biggest', circular='on')
            kmeans_partitions_empty.append(kmeans_partition[0])
            nmi_samples_1.append(normalized_mutual_info_score(true_labels, kmeans_partition[0]))


        print("NMI scores throught iterartions: {}".format(nmi_samples))
        print("KMeans partitions: {}".format(kmeans_partitions))
        nmi_scores.append(np.mean(nmi_samples))
        print("NMI score: {}".format(nmi_scores))

    plt.plot(nmi_scores)
    plt.xticks(list(range(len(sigma_list))), sigma_list)
    plt.title("{0} MRA partitioned by KMeans".format(MRA_type))
    plt.xlabel("Noise level (sigma)")
    plt.ylabel("NMI")
    plt.show()


sigma_list = np.array(range(10)) / 10
N = 100
runs_num = 10
plot_nmi_scores(N, sigma_list, runs_num, "Standard Normal")