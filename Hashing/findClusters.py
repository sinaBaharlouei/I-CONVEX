import csv
import networkx
from DataOperations import graphOperations
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

G = networkx.Graph()

with open('graph10K7', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        G.add_node(row[0])
        G.add_node(row[1])
        G.add_edge(row[0], row[1])

    clusters = graphOperations.find_connected_components(G)

    aList = list(clusters.values())

    aListCount = {}
    for i in aList:
        if i in aListCount:
            aListCount[i] += 1
        else:
            aListCount[i] = 1

    my_list = list(aListCount.values())
    print(len(my_list))

    print(my_list)
    num_bins = 100
    n, bins, patches = plt.hist(list(aListCount.values()), num_bins, facecolor='blue', alpha=0.5)
    plt.show()
