import csv
import timeit
from Bio import SeqIO
from collections import Counter
from igraph import *
import matplotlib.pyplot as plt


def read_fasta_file(fileName):
    return list(SeqIO.parse(fileName, 'fasta'))


write_mode = True
evaluation_mode = False
clusters = {}

dataset = read_fasta_file('reads.fasta')
id_dict = {}
id_list = []
G = Graph()

ind = 0
for item in dataset:
    id_dict[item.id] = ind
    ind += 1
    id_list.append(item.id)
    G.add_vertex(item.id)
    clusters[item.id] = -1

my_dict = []

ground_truth_cluster_dict = {}
ground_truth_predicted_dict = {}

edge_list = set()
with open('Net.csv', 'r') as csvfile:
    labels = list(csv.reader(csvfile, delimiter=','))

    with open('G.csv', 'r') as csvfile2:
        reader = list(csv.reader(csvfile2, delimiter=','))
        counter = 0
        for j in range(len(reader)):
            if float(labels[j][0]) == 1:
                edge_list.add((reader[j][0], reader[j][1]))


edge_list = list(edge_list)
G.add_edges(edge_list)
print("Graph is created successfully.")

# G.add_edges(edge_list)

print("Clustering:")
t1 = timeit.default_timer()
dendrogram = G.components()
t2 = timeit.default_timer()
print("clustering Time: ", t2 - t1)

members = dendrogram.membership
print(members)
most_common_index, count = Counter(members).most_common(1)[0]
print(most_common_index, count)
t20 = timeit.default_timer()
# print("max ", t20 - t2)

remove_list = []
for i in range(len(dendrogram)):
    if i != most_common_index:
        for item in dendrogram[i]:
            clusters[id_list[item]] = i + 1
            remove_list.append(id_list[item])

t3 = timeit.default_timer()
print("Finding the component with max size:", t3 - t2)


print("remove list length: ", len(remove_list))
G.delete_vertices(remove_list)
print(G.vs.get_attribute_values('name'))

print("Reclustering the biggest cluster:")
dendrogram = G.community_fastgreedy()

t4 = timeit.default_timer()
print("Community detection time: ", t4 - t3)
new_clusters = dendrogram.as_clustering()
# get the membership vector
membership = new_clusters.membership
print(membership)
vertices = G.vs.get_attribute_values('name')
for i in range(len(membership)):
    clusters[vertices[i]] = -membership[i]
print(clusters)

if write_mode:
    with open('FinalClusters.csv', 'wb') as f:  # Just use 'w' mode in 3.x
        print("Write to file ... ")
        w = csv. writer(f, delimiter=',')
        for key in clusters:
            w.writerow([key, clusters[key]])

t5 = timeit.default_timer()
print("Write time: ", t5 - t4)

if evaluation_mode:
    with open('unbalanced2MGT.csv', 'r') as csvfile:
        cluster_ids = list(csv.reader(csvfile, delimiter=','))

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
