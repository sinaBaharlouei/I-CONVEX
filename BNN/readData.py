import csv
import numpy as np
import random
import tensorflow as tf


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


def get_train_and_validation_tensors(filename, train_set_percentage):
    X = []
    Y = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        pairs = list(reader)
        pairs.pop(0)

    for item in pairs:
        pair = binarize_pair(item[0], item[1])
        X.append(pair)
        Y.append(int(item[2]))

    X = np.array(X)
    Y = np.array(Y)

    train_set_size = int(train_set_percentage * len(X))

    X_train = X[:train_set_size]
    X_validation = X[train_set_size:]

    Y_train = Y[:train_set_size]
    Y_validation = Y[train_set_size:]

    # input_layer = tf.reshape(X, [-1, 8, 400, 1])  # Width: 8, Height: 400
    # input_layer = tf.cast(input_layer, tf.float16)
    return X_train, Y_train, X_validation, Y_validation


# input_layer = tf.reshape(X, [-1, 400, 8, 1])

# print(X[0])
# print(Y[0])

"""
    for i in range(1, len(my_list)):
        pair = binarize_pair(my_list[i][0], my_list[i][1]) 
"""
