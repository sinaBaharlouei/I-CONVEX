import csv
import numpy as np


def binarize_sequence(sequence):
    w, h = len(sequence), 4

    letter_dictionary = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(sequence)):
        Matrix[letter_dictionary[sequence[i]]][i] = 1

    return np.array(Matrix)


def binarize_pair(sequence1, sequence2):
    w, h = len(sequence1), 8

    letter_dictionary = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(sequence1)):
        Matrix[letter_dictionary[sequence1[i]]][i] = 1
        Matrix[letter_dictionary[sequence2[i]] + 4][i] = 1

    return np.array(Matrix)


mat = binarize_pair("ATGCGGCATTA", "ATGCGGCATTA")
print(mat)

"""
with open('../similar50K.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    my_list = list(reader)
    for i in range(1, len(my_list)):
        X.append(my_list[i][0])
"""
