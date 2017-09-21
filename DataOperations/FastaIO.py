from Bio import SeqIO
import random
import csv


def read_fasta_file(fileName):
    return list(SeqIO.parse(fileName, 'fasta'))


def read_fasta_file_as_dict(fileName):
    return SeqIO.to_dict(SeqIO.parse(fileName, 'fasta'))


def write_LSH_matrix(file_name, LSH_dictionary):
    with open(file_name, 'w', newline='') as f:  # Just use 'w' mode in 3.x

        w = csv.writer(f, delimiter=',')
        for key in LSH_dictionary:
            w.writerow([key] + LSH_dictionary[key])


def write_buckets_to_csv(file_name, band_dictionary):
    with open(file_name, 'w') as f:  # Just use 'w' mode in 3.x

        w = csv.writer(f, delimiter=',')
        for key in band_dictionary:
            for item in band_dictionary[key]:
                print(key)
                print(item)
                w.writerow([key, item])


def write_graph_to_csv(file_name, G):
    with open(file_name, 'w', newline='') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f, delimiter=',')
        for edge in G.edges():
            w.writerow([edge[0], edge[1]])
