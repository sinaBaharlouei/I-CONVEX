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

dataset = FastaIO.read_fasta_file('../files/isoseq_flnc1.fasta')
dataset_dict = FastaIO.read_fasta_file_as_dict('../files/isoseq_flnc1.fasta')
id_dict = {}
id_list = []
G = networkx.Graph()
print(dataset_dict['m150803_002149_42161_c100745121910000001823165807071563_s1_p0/14/1140_57_CCS'].seq)

ind = 0
for item in dataset:
    id_dict[item.id] = ind
    ind += 1
    id_list.append(item.id)
    G.add_node(item.id)

print(id_list)
my_dict = []
"""
mat = scipy.io.loadmat('Data50.mat')
cluster_ids = mat['Id']
print(cluster_ids)
exit(0)
"""

cluster_ids = []

with open('result1.csv', 'r') as csvfile:
    reader = list(csv.reader(csvfile, delimiter=','))

    for item in reader:
        nex = item[2][4:]
        if item[2][0] == 'S':
            cluster_ids.append([int(nex)])
        else:
            cluster_ids.append([-1])

ground_truth_cluster_dict = {}
ground_truth_predicted_dict = {}

for item in cluster_ids:
    ground_truth_cluster_dict[item[0]] = []
    ground_truth_predicted_dict[item[0]] = []

for i in range(len(cluster_ids)):
    ground_truth_cluster_dict[cluster_ids[i][0]].append(i)


actual_pairs_set = set()
"""
mm = []
for key in ground_truth_cluster_dict:
    mm.append(len(ground_truth_cluster_dict[key]))
print(sorted(mm))
print(len(mm))
exit(0)
"""

for key in ground_truth_cluster_dict:
    current_cluster = list(ground_truth_cluster_dict[key])
    for i in range(1, len(current_cluster)):
        for j in range(i):
            actual_pairs_set.add((current_cluster[i], current_cluster[j]))

# print(actual_pairs_set)
# print(len(actual_pairs_set))
# print(id_dict)

with open('MHNET10.csv', 'r') as csvfile:
    labels = list(csv.reader(csvfile, delimiter=','))

    with open('MGraphK15R1B10P10.csv', 'r') as csvfile2:
        reader = list(csv.reader(csvfile2, delimiter=','))
        counter = 0
        for i in range(len(reader)):
            if float(labels[i][0]) == 1:
                G.add_edge(reader[i][0], reader[i][1])

        print('Clustering ...')
        # clusters = community.best_partition(G)
        clusters = graphOperations.find_connected_components(G)

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
            if aListCount[key] < 4:
                merged_clusters.append(key)

        s = 0
        for node in clusters:
            if clusters[node] in merged_clusters:
                clusters[node] = -1
                s += 1

        nodesClusterList = list(clusters.values())
        aListCount = {}
        for i in nodesClusterList:
            if i in aListCount:
                aListCount[i] += 1
            else:
                aListCount[i] = 1

        my_list = list(aListCount.values())
        print("my_list:", sorted(my_list, reverse=True))
        print(len(my_list))

        #
        clusterRatio = {}
        clusterLength = {}
        for cluster in ground_truth_cluster_dict:
            for item in ground_truth_cluster_dict[cluster]:  # all reads that are in the same cluster
                predicted_cluster = clusters[id_list[item]]
                ground_truth_predicted_dict[cluster].append(predicted_cluster)

            list_counter = Counter(ground_truth_predicted_dict[cluster])
            max_repetition = list_counter.most_common(1)[0][1]

            clusterRatio[cluster] = max_repetition / len(ground_truth_predicted_dict[cluster])
            clusterLength[cluster] = len(ground_truth_cluster_dict[cluster])
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
        plt.xlabel("Histogram of Clusters Sizes")
        plt.title("Size of Detected Clusters")
        plt.ylabel("Frequency")
        plt.hist(list(aListCount.values()), num_bins, facecolor='blue', alpha=0.5)
        plt.show()
        # print(aListCount)

        # Histogram for accuracy of clustering for each read (Proportion of actual neighbors clustered in the same bucket with the read)
        plt.xlabel("Histogram of Detected Neighbors Ratio")
        plt.title("Detected Neighbors ratio")
        plt.ylabel("Frequency")
        plt.hist(my_dict, num_bins, facecolor='blue', alpha=0.5)
        plt.show()

        # Histogram for actual clusters
        plt.xlabel("Histogram of Clusters Ratio")
        plt.title("Clusters Ratio")
        plt.ylabel("Frequency")
        plt.hist(list(clusterRatio.values()), num_bins, facecolor='blue', alpha=0.5)
        plt.show()

        for key in clusterRatio:
            print(clusterRatio[key], clusterLength[key])