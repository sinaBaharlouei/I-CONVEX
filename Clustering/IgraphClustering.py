import csv
import timeit

import networkx
from DataOperations import graphOperations
import matplotlib.pyplot as plt
import community
from DataOperations import FastaIO
from collections import Counter

print("sal")
dataset = FastaIO.read_fasta_file('unbalanced700K.fasta')
id_dict = {}
id_list = []
G = networkx.Graph()
print("hi")
ind = 0
for item in dataset:
    id_dict[item.id] = ind
    ind += 1
    id_list.append(item.id)
    G.add_node(item.id)

my_dict = []

"""
mat = scipy.io.loadmat('Data50.mat')
cluster_ids = mat['Id']

"""

print("Sina")
with open('unbalanced700KGT.csv', 'r') as csvfile:
    cluster_ids = list(csv.reader(csvfile, delimiter=','))


print("OK")
ground_truth_cluster_dict = {}
ground_truth_predicted_dict = {}

for item in cluster_ids:
    ground_truth_cluster_dict[int(item[0])] = []
    ground_truth_predicted_dict[int(item[0])] = []

print(cluster_ids)

for i in range(len(cluster_ids)):
    ground_truth_cluster_dict[int(cluster_ids[i][0])].append(i)

# print(ground_truth_cluster_dict)
# exit(0)
write_mode = 1
"""
with open('Net50.csv', 'r') as csvfile:
    labels = list(csv.reader(csvfile, delimiter=','))

    with open('D50MGK15R1B10P10.csv', 'r') as csvfile2:
        reader = list(csv.reader(csvfile2, delimiter=','))
        counter = 0
        for i in range(len(reader)):
            if float(labels[i][0]) == 1:
                G.add_edge(reader[i][0], reader[i][1])
"""

for i in range(1, 7):
    with open('Net' + str(i) + '.csv', 'r') as csvfile:
        labels = list(csv.reader(csvfile, delimiter=','))

        with open('G' + str(i), 'r') as csvfile2:
            reader = list(csv.reader(csvfile2, delimiter=','))
            counter = 0
            for j in range(len(reader)):
                if float(labels[j][0]) == 1:
                    G.add_edge(reader[j][0], reader[j][1])

print('Clustering ...')
t1 = timeit.default_timer()
# clusters = community.best_partition(G)
clusters = graphOperations.find_connected_components(G)
print("Connected Components has found.")
if write_mode == 1:
    most_common_index, count = Counter(clusters.values()).most_common(1)[0]
    remove_list = []
    print(most_common_index, count)
    for key in clusters:
        if clusters[key] != most_common_index:
            remove_list.append(key)

    print("remove list length: ", len(remove_list))
    G.remove_nodes_from(remove_list)
    print("Community detection ...")
    new_clusters = community.best_partition(G)
    max_index = 0
    print("Merging ....")
    for key in new_clusters:
        if new_clusters[key] > max_index:
            max_index = new_clusters[key]
        new_clusters[key] = - new_clusters[key]

    for key in new_clusters:
        clusters[key] = new_clusters[key]

    for key in clusters:
        clusters[key] += max_index
    t2 = timeit.default_timer()
    print(t2 - t1)

    nodesClusterList = list(clusters.values())
    print("hi1")
    aListCount = {}
    for i in nodesClusterList:
        if i in aListCount:
            aListCount[i] += 1
        else:
            aListCount[i] = 1

    print("hi2")
    merged_clusters = []
    # Merge clusters with one element

    for key in aListCount:
        if aListCount[key] < 5:
            merged_clusters.append(key)
    print("hi3")
    s = 0
    for node in clusters:
        if clusters[node] in merged_clusters:
            clusters[node] = -1
            s += 1
    print("hi4")
    with open('Final700KClusters.csv', 'w', newline='') as f:  # Just use 'w' mode in 3.x
        print("Write to file ... ")
        w = csv.writer(f, delimiter=',')
        for key in clusters:
            w.writerow([key, clusters[key]])

nodesClusterList = list(clusters.values())
print("hi1")
aListCount = {}
for i in nodesClusterList:
    if i in aListCount:
        aListCount[i] += 1
    else:
        aListCount[i] = 1

print("hi2")
merged_clusters = []
# Merge clusters with one element

for key in aListCount:
    if aListCount[key] < 5:
        merged_clusters.append(key)
print("hi3")
s = 0
for node in clusters:
    if clusters[node] in merged_clusters:
        clusters[node] = -1
        s += 1
print("hi4")
nodesClusterList = list(clusters.values())
aListCount = {}
for i in nodesClusterList:
    if i in aListCount:
        aListCount[i] += 1
    else:
        aListCount[i] = 1
print("hi5")
x = 0
my_list = list(aListCount.values())
for item in my_list:
    if item == 1:
        x += 1
print(x)
sorted_clusters = sorted(my_list, reverse=True)

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
    actual_cluster_id = int(cluster_ids[id_dict[key]][0])
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

num_bins = 10
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

print(list(aListCount.values()))
print(len(list(aListCount.values())))
