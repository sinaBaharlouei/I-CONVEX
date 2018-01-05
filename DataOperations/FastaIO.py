from Bio import SeqIO
import random
import csv
import numpy as np


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


def binarize_sequence(sequence):
    w, h = len(sequence), 4

    letter_dictionary = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(sequence)):
        Matrix[letter_dictionary[sequence[i]]][i] = 1

    return np.array(Matrix)


def binarize_pair(sequence1, sequence2):
    # w, h = 98, 8
    w, h = 400, 8

    letter_dictionary = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Matrix = np.zeros(shape=(h, w))
    for i in range(w):
        Matrix[letter_dictionary[sequence1[i]]][i] = 1
        Matrix[letter_dictionary[sequence2[i]] + 4][i] = 1

    return Matrix


def read_next_batch(similarity_graph, fasta_file, batch_size):
    reads_dict = read_fasta_file_as_dict(fasta_file)

    with open(similarity_graph, 'r') as csvfile:
        pairs = list(csv.reader(csvfile, delimiter=','))

        number_of_pairs = len(pairs)
        number_of_batches = number_of_pairs // batch_size

        for i in range(number_of_batches):
            current_batch = []
            start = number_of_batches * i
            end = start + batch_size

            for j in range(start, end):
                current_pair = pairs[j]
                seq1 = reads_dict[current_pair[0]].seq
                seq2 = reads_dict[current_pair[1]].seq
                binarized = binarize_pair(seq1, seq2)
                current_batch.append(binarized)
            yield np.array(current_batch)


        # Last batch
        current_batch = []
        start = number_of_batches * batch_size
        end = number_of_pairs

        for j in range(start, end):
            current_pair = pairs[j]
            seq1 = reads_dict[current_pair[0]].seq
            seq2 = reads_dict[current_pair[1]].seq
            binarized = binarize_pair(seq1, seq2)
            current_batch.append(binarized)
        yield np.array(current_batch)
