import csv
import numpy as np
import collections


class DataSet(object):
    """Dataset class object."""

    def __init__(self,
                 pairs,
                 labels
                 ):
        self._pairs = pairs
        self._num_examples = pairs.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def pairs(self):
        return self._pairs

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._pairs = self._pairs[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._pairs[start:end], self._labels[start:end]


def read_data_sets(filename,
                   validation_percentage,
                   test_percentage,
                   ):
    X = []
    Y = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        pairs = list(reader)
        pairs.pop(0)

    for item in pairs:
        pair = binarize_pair(item[0], item[1])
        X.append(pair)
        Y.append(2 * int(item[2]) - 1)

    all_pairs = np.array(X)
    print("pairs shape:", all_pairs.shape)
    all_labels = np.array(Y)
    print("all_labels_shape:", all_labels.shape)

    """Set the pairs and labels."""
    num_training = int((1 - (validation_percentage + test_percentage)) * len(X))
    num_validation = int(validation_percentage * len(X))
    num_test = int(test_percentage * len(X))

    all_pairs = all_pairs.reshape(all_pairs.shape[0],
                                  all_pairs.shape[1], all_pairs.shape[2], 1)
    #all_labels = dense_to_one_hot(all_labels, len(all_labels))

    mask = range(num_training)
    train_pairs = all_pairs[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_validation)
    validation_pairs = all_pairs[mask]
    validation_labels = all_labels[mask]

    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    test_images = all_pairs[mask]
    test_labels = all_labels[mask]


    train = DataSet(train_pairs, train_labels)
    validation = DataSet(validation_pairs, validation_labels)
    test = DataSet(test_images, test_labels)

    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


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

    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(w):
        Matrix[letter_dictionary[sequence1[i]]][i] = 1
        Matrix[letter_dictionary[sequence2[i]] + 4][i] = 1

    return np.array(Matrix)

# input_layer = tf.reshape(X, [-1, 400, 8, 1])
