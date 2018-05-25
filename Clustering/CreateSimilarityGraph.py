import csv
import networkx as nx
from DataOperations import FastaIO
import timeit

# Collect the matrix
LSH_dict = {}
bands_dict = {}

with open('hash_functions.csv', 'r') as csvfile:
    pairs = list(csv.reader(csvfile, delimiter=','))
    r = int(pairs[0][0])
    b = int(pairs[0][1])
    number_of_chunks = int(pairs[0][2])


t0 = timeit.default_timer()
for i in range(number_of_chunks):
    with open('LSH_' + str(i+1) + '.csv') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=','))
        print(rows[2][0])
        j = 0
        for row in rows:
            if j % 2 == 0:
                row_len = len(row)
                LSH_dict[row[0]] = []
                for k in range(b):
                    bands_dict[(k+1, row[k+1])] = []  # keys: (column, value)
                    LSH_dict[row[0]].append(row[k+1])
            j += 1

t1 = timeit.default_timer()
print(t1-t0)

# Create graph
G = nx.Graph()
for key in LSH_dict:
    G.add_node(key)
    band_values = LSH_dict[key]
    for i in range(len(band_values)):
        bands_dict[(i+1, band_values[i])].append(key)


number_of_permutations = 10

for key in bands_dict:
    L = len(bands_dict[key])
    t = 1
    while L > t and t < number_of_permutations:
        for i in range(L-t):
            G.add_edge(bands_dict[key][i], bands_dict[key][i+t])
        t += 1

t2 = timeit.default_timer()
print(t2 - t1)
print(bands_dict)
exit(0)
FastaIO.write_graph_to_csv('MinGraphK' + str(15) + 'R' + str(r) + 'B' + str(b) + 'P' + str(number_of_permutations) + '.csv', G)
t3 = timeit.default_timer()
print(t3 - t2)


print(len(G.edges()))
