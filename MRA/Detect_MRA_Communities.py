from cdlib import algorithms as cd
import MRA_Graphs
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_cd
import matplotlib.pyplot as plt
import igraph as ig
import Kmeans
import time

def extract_communities_list(communities_list, N):
    community_labels = np.zeros(N)
    for comm_index, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            community_labels[node] = comm_index
    return community_labels

def calculate_snr(x, noise, L):
    Psignal = (1/L)*np.sum(np.power(x[0],2))
    Pnoise = (1/L)*np.sum(np.power(noise[0],2))
    if Pnoise == 0: Pnoise = 0.001 # Configure low noise to get finite SNR
    snr = Psignal/Pnoise

    return 10*np.log10(snr)

def plot_nmi_scores(N, sigma_list, samples_num, MRA_type):
    nmi_scores = {"leiden": [], "louvain": [], "leading_eigenvector": [], "spinglass": [],
                  "walktrap": [], "infomap": [], "greedy": [], "LPA": [], "betweenness": [], "kmeans": []}
    time_scores = {"leiden": [], "louvain": [], "leading_eigenvector": [], "spinglass": [],
                  "walktrap": [], "infomap": [], "greedy": [], "LPA": [], "betweenness": [], "kmeans": []}
    snr_list = []
    for sigma in sigma_list:
        nmi_samples = {"leiden": [], "louvain": [], "leading_eigenvector": [], "spinglass": [],
                       "walktrap": [], "infomap": [], "greedy": [], "LPA": [], "betweenness": [], "kmeans": []}
        time_samples = {"leiden": [], "louvain": [], "leading_eigenvector": [], "spinglass": [],
                       "walktrap": [], "infomap": [], "greedy": [], "LPA": [], "betweenness": [], "kmeans": []}
        snr_samples = []

        print("sigma: {}".format(sigma))

        # Calculate NMI score for a number (samples_num) of random graphs
        for i in range(samples_num):
            # Generate appropriate MRA graph
            if MRA_type == "Rect_Trian":
                K=2
                nx_G, ig_G, true_labels, x, noise, y = MRA_Graphs.MRA_Rect_Trian(N, L=50, K=K, sigma=sigma)
                snr_samples.append(calculate_snr(x, noise, 50))
            elif MRA_type == "Standard_Normal":
                K=2
                nx_G, ig_G, true_labels, x, noise, y = MRA_Graphs.MRA_StandardNormal(N, L=50, K=K, sigma=sigma)
                snr_samples.append(calculate_snr(x, noise, 50))
            else:
                K=2
                nx_G, ig_G, true_labels, x, noise, y = MRA_Graphs.MRA_CorrelatedNormal(N, L=51, K=K, a=1, b=2, choice=1, sigma=sigma)
                snr_samples.append(calculate_snr(x, noise, 51))

            print("true: {}".format(true_labels))

            start = time.time()
            kmeans_labels = Kmeans.kmeans(y, K)
            end = time.time()
            time_samples["kmeans"].append(end - start)
            print("kmeans: {}".format(kmeans_labels))
            nmi_samples["kmeans"].append(normalized_mutual_info_score(true_labels, kmeans_labels))

            start = time.time()
            leiden = cd.leiden(nx_G, weights='weight')
            end = time.time()
            time_samples["leiden"].append(end-start)
            leiden_labels = extract_communities_list(leiden.communities, N)
            print("leiden: {}".format(leiden_labels))
            nmi_samples["leiden"].append(normalized_mutual_info_score(true_labels, leiden_labels))

            start = time.time()
            louvain = cd.louvain(nx_G, weight='weight')
            end = time.time()
            time_samples["louvain"].append(end - start)
            louvain_labels = extract_communities_list(louvain.communities, N)
            print("louvain: {}".format(louvain_labels))
            nmi_samples["louvain"].append(normalized_mutual_info_score(true_labels, louvain_labels))

            # Convert NetworkX graph to Igraph graph
            #G = ig.Graph.from_networkx(G)

            #start = time.time()
            #infomap = ig.Graph.community_infomap(ig_G, edge_weights='weight')
            #end = time.time()
            #time_samples["infomap"].append(end - start)
            #print("infomap: {}".format(infomap.membership))
            #nmi_samples["infomap"].append(normalized_mutual_info_score(true_labels, infomap.membership))

            start = time.time()
            greedy = ig.Graph.community_fastgreedy(ig_G, weights='weight')
            end = time.time()
            time_samples["greedy"].append(end - start)
            greedy = greedy.as_clustering()
            print("greedy: {}".format(greedy.membership))
            nmi_samples["greedy"].append(normalized_mutual_info_score(true_labels, greedy.membership))

            #start = time.time()
            #LPA = ig.Graph.community_label_propagation(ig_G, weights='weight')
            #end = time.time()
            #time_samples["LPA"].append(end - start)
            #print("LPA: {}".format(LPA.membership))
            #nmi_samples["LPA"].append(normalized_mutual_info_score(true_labels, LPA.membership))

            #start = time.time()
            #betweenness = ig.Graph.community_edge_betweenness(ig_G, weights='weight')
            #end = time.time()
            #time_samples["betweenness"].append(end - start)
            #betweenness = betweenness.as_clustering()
            #print("betweenness: {}".format(betweenness.membership))
            #nmi_samples["betweenness"].append(normalized_mutual_info_score(true_labels, betweenness.membership))

            start = time.time()
            leading_eigenvector = ig.Graph.community_leading_eigenvector(ig_G, weights="weight")
            end = time.time()
            time_samples["leading_eigenvector"].append(end - start)
            print("leading eigenvector: {}".format(leading_eigenvector.membership))
            nmi_samples["leading_eigenvector"].append(normalized_mutual_info_score(true_labels, leading_eigenvector.membership))

            #multilevel = ig.Graph.community_multilevel(G, weights="weight")
            #print("multilevel: {}".format(multilevel.membership))
            #nmi_samples["multilevel"].append(normalized_mutual_info_score(true_labels, multilevel.membership))

            #start = time.time()
            #spinglass = ig.Graph.community_spinglass(ig_G, weights="weight")
            #end = time.time()
            #time_samples["spinglass"].append(end - start)
            #print("spinglass: {}".format(spinglass.membership))
            #nmi_samples["spinglass"].append(normalized_mutual_info_score(true_labels, spinglass.membership))

            start = time.time()
            walktrap = ig.Graph.community_walktrap(ig_G, weights="weight")
            end = time.time()
            time_samples["walktrap"].append(end - start)
            walktrap = walktrap.as_clustering()
            print("walktrap: {}".format(walktrap.membership))
            nmi_samples["walktrap"].append(normalized_mutual_info_score(true_labels, walktrap.membership))

            # Set NMI score for each algrorithm
        #nmi_scores["infomap"].append(np.mean(nmi_samples["infomap"]))
        nmi_scores["greedy"].append(np.mean(nmi_samples["greedy"]))
        #nmi_scores["LPA"].append(np.mean(nmi_samples["LPA"]))
        #nmi_scores["betweenness"].append(np.mean(nmi_samples["betweenness"]))
        nmi_scores["kmeans"].append(np.mean(nmi_samples["kmeans"]))
        nmi_scores["leiden"].append(np.mean(nmi_samples["leiden"]))
        nmi_scores["louvain"].append(np.mean(nmi_samples["louvain"]))
        nmi_scores["leading_eigenvector"].append(np.mean(nmi_samples["leading_eigenvector"]))
        #nmi_scores["multilevel"].append(np.mean(nmi_samples["multilevel"]))
        #nmi_scores["spinglass"].append(np.mean(nmi_samples["spinglass"]))
        nmi_scores["walktrap"].append(np.mean(nmi_samples["walktrap"]))

        #time_scores["infomap"].append(np.round(np.mean(time_samples["infomap"]), 2))
        time_scores["greedy"].append(np.round(np.mean(time_samples["greedy"]), 2))
        #time_scores["LPA"].append(np.round(np.mean(time_samples["LPA"]), 2))
        #time_scores["betweenness"].append(np.round(np.mean(time_samples["betweenness"]), 2))
        time_scores["kmeans"].append(np.round(np.mean(time_samples["kmeans"]), 2))
        time_scores["leiden"].append(np.round(np.mean(time_samples["leiden"]), 2))
        time_scores["louvain"].append(np.round(np.mean(time_samples["louvain"]), 2))
        time_scores["leading_eigenvector"].append(np.mean(time_samples["leading_eigenvector"]))
        # nmi_scores["multilevel"].append(np.mean(nmi_samples["multilevel"]))
        #time_scores["spinglass"].append(np.round(np.mean(time_samples["spinglass"]), 2))
        time_scores["walktrap"].append(np.round(np.mean(time_samples["walktrap"]), 2))

        snr_list.append(np.mean(snr_samples))

    #plt.plot(np.flip(time_scores["infomap"]), color='#575757',
     #          marker='o', mfc='#f1362b', mec='#f1362b' ,label='infomap')
    plt.plot(np.flip(time_scores["greedy"]), color='#575757',
               marker='x', mfc='#17c436', mec='#17c436', label='greedy')
    #plt.plot(np.flip(time_scores["LPA"]), color='#575757',
    #           marker='o', mfc='#d9a000', mec='#d9a000', label='LPA')
    #plt.plot(np.flip(time_scores["betweenness"]), color='#575757',
    #           marker='o', mfc='#532ce9', mec='#532ce9', label='betweenness')
    plt.plot(np.flip(time_scores["kmeans"]), color='#575757',
                        marker='o', mfc='#d9a000', mec='#d9a000', label='kmeans', linewidth=3)
    plt.plot(np.flip(time_scores["leiden"]), color='#575757',
               marker='x', mfc='#532ce9', mec='#532ce9', label='leiden')
    plt.plot(np.flip(time_scores["louvain"]), color='#575757',
               marker='x', mfc='#f1362b', mec='#f1362b', label='louvain')
    plt.plot(np.flip(time_scores["leading_eigenvector"]), color='#575757',
               marker='x', mfc='#d9a000', mec='#d9a000', label='leading eigenvector')
    plt.plot(np.flip(time_scores["walktrap"]), color='#575757',
               marker='x', mfc='#02B789', mec='#02B789', label='walktrap')
    #plt.plot(np.flip(time_scores["spinglass"]), color='#575757',
    #           marker='o', mfc='#02B789', mec='#02B789', label='spinglass')
    plt.xticks(list(range(len(snr_list))), np.flip(np.floor(snr_list).astype('int')))
    plt.title("{} MRA\n (K={}, Signal Length={}, Runs Number={})".format(MRA_type, K, 50, 10))
    plt.xlabel("SNR[dB]")
    plt.ylabel("Time[s]")
    plt.legend()
    plt.grid()
    plt.show()

    #plt.plot(np.flip(nmi_scores["infomap"]), color='#575757',
    #           marker='o', mfc='#f1362b', mec='#f1362b', label="infomap")
    plt.plot(np.flip(nmi_scores["greedy"]), color='#575757',
               marker='x', mfc='#17c436', mec='#17c436', label="greedy")
    #plt.plot(np.flip(nmi_scores["LPA"]), color='#575757',
    #           marker='o', mfc='#d9a000', mec='#d9a000', label="LPA")
    #plt.plot(np.flip(nmi_scores["betweenness"]), color='#575757',
    #           marker='o', mfc='#532ce9', mec='#532ce9', label="betweenness")
    plt.plot(np.flip(nmi_scores["kmeans"]), color='#575757',
             marker='o', mfc='#d9a000', mec='#d9a000', label='kmeans', linewidth=3)
    plt.plot(np.flip(nmi_scores["leiden"]), color='#575757',
               marker='x', mfc='#532ce9', mec='#532ce9', label="leiden")
    plt.plot(np.flip(nmi_scores["louvain"]), color='#575757',
               marker='x', mfc='#f1362b', mec='#f1362b', label="louvain")
    plt.plot(np.flip(nmi_scores["leading_eigenvector"]), color='#575757',
               marker='x', mfc='#d9a000', mec='#d9a000', label="leading eigenvector")
    #plt.plot(np.flip(nmi_scores["multilevel"]), label="multilevel")
    plt.plot(np.flip(nmi_scores["walktrap"]), color='#575757',
               marker='x', mfc='#02B789', mec='#02B789', label="walktrap")
    #plt.plot(np.flip(nmi_scores["spinglass"]), color='#575757',
    #          marker='x', mfc='#02B789', mec='#02B789', label="spinglass")
    #plt.xscale("symlog", linthresh=np.mean(np.abs(snr_list)))
    #plt.xscale("symlog", linthresh=0.1)
    plt.xticks(list(range(len(snr_list))), np.flip(np.floor(snr_list).astype('int')))
    plt.title("{} MRA\n (K={}, Signal Length={}, Runs Number={})".format(MRA_type,K,50,10))
    plt.xlabel("SNR[dB]")
    plt.ylabel("NMI")
    plt.legend()
    plt.grid()
    plt.show()

sigma_list = np.linspace(0, 0.5, 10)
N = 1000
samples_num = 10

plot_nmi_scores(N, sigma_list, samples_num, "Rect_Trian")
#plot_nmi_scores(N, sigma_list, samples_num, "Standard_Normal")
#plot_nmi_scores(N, sigma_list, samples_num, "Correlated_Normal")