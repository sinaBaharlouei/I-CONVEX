import os
import csv
from Bio import SeqIO


def read_fasta_file_as_dict(fileName):
    return SeqIO.to_dict(SeqIO.parse(fileName, 'fasta'))


if not os.path.exists('Clusters'):
    os.makedirs('Clusters')


fasta_dict = read_fasta_file_as_dict("reads.fasta")

cluster_dictionary = {}
with open('MergedClusters.csv', 'r') as csvfile:
    pairs = list(csv.reader(csvfile, delimiter=','))
    for item in pairs:
        cluster_dictionary[int(item[1])] = []

    print(cluster_dictionary.keys())
    print(len(cluster_dictionary.keys()))

    for item in pairs:
        cluster_dictionary[int(item[1])].append(item[0])

    print("Hi")
    j = 0
    for key in cluster_dictionary:
        records = []
        os.makedirs('Clusters/' + str(key))
        for item in cluster_dictionary[key]:
            records.append(fasta_dict[item])

        SeqIO.write(records, 'Clusters/' + str(key) + '/final_cluster.fasta', "fasta")
        j += 1
        if j % 20 == 0:
            print(j)