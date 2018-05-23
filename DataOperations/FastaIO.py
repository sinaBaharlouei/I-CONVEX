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


def binarize_sequence2(sequence):
    w, h = len(sequence), 4

    letter_dictionary = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Matrix = [[0 for x in range(400)] for y in range(h)]
    if w >= 400:
        for i in range(400):
            Matrix[letter_dictionary[sequence[i]]][i] = 1

    else:
        for i in range(w):
            Matrix[letter_dictionary[sequence[i]]][i] = 1

        for i in range(w, 400):
            # Add a to the rest
            Matrix[0][i] = 1
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

    # Converts to the right format
    one_hot_encoded_dict = {}
    for key in reads_dict:
        one_hot_encoded_dict[key] = binarize_sequence2(reads_dict[key].seq)

    with open(similarity_graph, 'r') as csvfile:
        pairs = list(csv.reader(csvfile, delimiter=','))

        number_of_pairs = len(pairs)
        number_of_batches = number_of_pairs // batch_size

        for i in range(number_of_batches):
            current_batch = []
            start = batch_size * i
            end = start + batch_size

            for j in range(start, end):
                current_pair = pairs[j]
                seq1 = one_hot_encoded_dict[current_pair[0]]
                seq2 = one_hot_encoded_dict[current_pair[1]]
                current_batch.append(np.concatenate((seq1, seq2)))
                """
                seq1 = reads_dict[current_pair[0]].seq
                seq2 = reads_dict[current_pair[1]].seq
                if len(seq1) < 400:
                    rest = ''
                    for k in range(400 - len(seq1)):
                        rest += 'A'
                    seq1 += rest

                if len(seq2) < 400:
                    rest = ''
                    for k in range(400 - len(seq2)):
                        rest += 'A'
                    seq2 += rest

                seq1 = seq1[0:400]
                seq2 = seq2[0:400]
                binarized = binarize_pair(seq1, seq2)
                """
            yield np.array(current_batch), number_of_batches, batch_size
            print(start, end)

        # Last batch
        current_batch = []
        start = number_of_batches * batch_size
        end = number_of_pairs
        batch_size = end - start

        for j in range(start, end):
            current_pair = pairs[j]
            seq1 = one_hot_encoded_dict[current_pair[0]]
            seq2 = one_hot_encoded_dict[current_pair[1]]
            current_batch.append(np.concatenate((seq1, seq2)))
            """
            seq1 = reads_dict[current_pair[0]].seq
            seq2 = reads_dict[current_pair[1]].seq
            if len(seq1) < 400:
                rest = ''
                for k in range(400 - len(seq1)):
                    rest += 'A'
                seq1 += rest

            if len(seq2) < 400:
                rest = ''
                for k in range(400 - len(seq2)):
                    rest += 'A'
                seq2 += rest

            seq1 = seq1[0:400]
            seq2 = seq2[0:400]

            binarized = binarize_pair(seq1, seq2)
            current_batch.append(binarized)
            """
        yield np.array(current_batch), number_of_batches, batch_size


def split_file(filename, number_of_chunks):
    fasta_file = read_fasta_file(filename)
    number_of_reads = len(fasta_file)

    batch_size = number_of_reads // number_of_chunks

    for i in range(number_of_chunks - 1):
        start = i * batch_size
        end = start + batch_size

        sequences = []
        for j in range(start, end):
            sequences.append(fasta_file[j])

        SeqIO.write(sequences, "chunk" + str(i+1) + ".fasta", "fasta")

    # last file
    start = (number_of_chunks - 1) * batch_size
    end = number_of_reads
    sequences = []
    for j in range(start, end):
        sequences.append(fasta_file[j])

    SeqIO.write(sequences, "chunk" + str(number_of_chunks) + ".fasta", "fasta")


# split_file('../files/reads50.fasta', 10)
# x = read_fasta_file('chunk10.fasta')
# print(len(x))
