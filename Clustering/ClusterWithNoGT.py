import csv
import timeit

import networkx
from DataOperations import graphOperations
import community
from DataOperations import FastaIO
from collections import Counter

dataset = FastaIO.read_fasta_file('reads1M.fasta')
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

for i in range(1, 5):
    with open('Net' + str(i) + '.csv', 'r') as csvfile:
        labels = list(csv.reader(csvfile, delimiter=','))

        with open('G' + str(i) + '.csv', 'r') as csvfile2:
            reader = list(csv.reader(csvfile2, delimiter=','))
            counter = 0
            for j in range(len(reader)):
                if float(labels[j][0]) == 1:
                    G.add_edge(reader[j][0], reader[j][1])

print('Clustering ...')
t1 = timeit.default_timer()
# clusters = community.best_partition(G)
clusters = graphOperations.find_connected_components(G)

nodesClusterList = list(clusters.values())
print("hi1")
aListCount = {}
for i in nodesClusterList:
    if i in aListCount:
        aListCount[i] += 1
    else:
        aListCount[i] = 1

s = 0
k = 0
for item in aListCount:
    if aListCount[item] > 10:
        print(item, aListCount[item])
        s += aListCount[item]
    elif aListCount[item] < 4:
        k += aListCount[item]
print(s)
print(k)
exit(0)

print("Connected Components has found.")
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

print(aListCount)
exit(0)
print("hi2")
merged_clusters = []
# Merge clusters with one element

for key2 in aListCount:
    if aListCount[key2] < 10:
        merged_clusters.append(key2)
print("hi3")
s = 0

for node in clusters:
    if clusters[node] in merged_clusters:
        clusters[node] = -1
        s += 1
print("hi4")

with open('finalML.csv', 'w', newline='') as f:  # Just use 'w' mode in 3.x
    print("Write to file ... ")
    w = csv.writer(f, delimiter=',')
    for key2 in clusters:
        w.writerow([key2, clusters[key2]])
exit(0)
