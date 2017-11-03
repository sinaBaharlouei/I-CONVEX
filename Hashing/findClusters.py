import csv
import networkx
from DataOperations import graphOperations
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.io
import community
from DataOperations import FastaIO
from collections import Counter

dataset = FastaIO.read_fasta_file('../files/reads50.fasta')
id_dict = {}
id_list = []
G = networkx.Graph()

ind = 0
for item in dataset:
    id_dict[item.id] = ind
    ind += 1
    id_list.append(item.id)
    G.add_node(item.id)

my_dict = []
mat = scipy.io.loadmat('Data50.mat')
cluster_ids = mat['Id']

ground_truth_cluster_dict = {}
ground_truth_predicted_dict = {}

for item in cluster_ids:
    ground_truth_cluster_dict[item[0]] = []
    ground_truth_predicted_dict[item[0]] = []

for i in range(len(cluster_ids)):
    ground_truth_cluster_dict[cluster_ids[i][0]].append(i)

print(ground_truth_cluster_dict)

# predicted_value_for each cluster


with open('wholeGraphK9R3B9.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        G.add_edge(row[0], row[1])

    clusters = community.best_partition(G)
    # clusters = graphOperations.find_connected_components(G)
    nodesClusterList = list(clusters.values())

    aListCount = {}
    for i in nodesClusterList:
        if i in aListCount:
            aListCount[i] += 1
        else:
            aListCount[i] = 1

    merged_clusters = []
    # Merge clusters with one element
    for key in aListCount:
        if aListCount[key] < 2:
            merged_clusters.append(key)
    print(merged_clusters)

    for node in clusters:
        if clusters[node] in merged_clusters:
            clusters[node] = -1

    nodesClusterList = list(clusters.values())
    aListCount = {}
    for i in nodesClusterList:
        if i in aListCount:
            aListCount[i] += 1
        else:
            aListCount[i] = 1

    my_list = list(aListCount.values())

    clusterRatio = {}
    for cluster in ground_truth_cluster_dict:
        for item in ground_truth_cluster_dict[cluster]:
            predicted_cluster = nodesClusterList[item]
            ground_truth_predicted_dict[cluster].append(predicted_cluster)

        list_counter = Counter(ground_truth_predicted_dict[cluster])
        max_repetition = list_counter.most_common(1)[0][1]
        clusterRatio[cluster] = max_repetition / len(ground_truth_predicted_dict[cluster])

    # print(ground_truth_predicted_dict)
    # print(clusterRatio)

    predicted_cluster_dict = {}  # key is cluster id and for each cluster id we have all nodes in the cluster
    for key in clusters:
        predicted_cluster_dict[clusters[key]] = []

    for key in clusters:
        predicted_cluster_dict[clusters[key]].append(id_dict[key])

    # print(predicted_cluster_dict)
    one_index = 0
    for key in clusters:

        predicted_cluster = clusters[key]

        # if my_list[predicted_cluster] < 2:
        #    continue

        # print(predicted_cluster)
        # print(id_dict[key])
        actual_cluster_id = cluster_ids[id_dict[key]][0]
        # print(actual_cluster_id)
        actual_neighbors = ground_truth_cluster_dict[actual_cluster_id]
        if len(actual_neighbors) <= 1:
            my_dict.append(1)
            continue

        # print(actual_neighbors)
        # print("-------------------")
        predicted_true = 0
        total = 0
        for actual_neighbor in actual_neighbors:
            total += 1
            if predicted_cluster == clusters[id_list[actual_neighbor]]:
                predicted_true += 1

        my_dict.append(predicted_true / total)

    num_bins = 20
    # Histogram for size of the clusters
    plt.hist(list(aListCount.values()), num_bins, facecolor='blue', alpha=0.5)
    plt.show()
    print(aListCount)

    # Histogram for accuracy of clustering for each read (Proportion of actual neighbors clustered in the same bucket with the read)
    plt.hist(my_dict, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

    # Histogram for actual clusters
    plt.hist(list(clusterRatio.values()), num_bins, facecolor='blue', alpha=0.5)
    plt.show()
