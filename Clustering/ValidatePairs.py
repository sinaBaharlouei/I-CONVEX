import csv
import tensorflow as tf
import numpy as np
import timeit
from Bio import SeqIO


def read_fasta_file_as_dict(fileName):
    return SeqIO.to_dict(SeqIO.parse(fileName, 'fasta'))


def binarize_sequence2(sequence):
    width, h = len(sequence), 4

    letter_dictionary = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    Matrix = [[0 for x in range(400)] for y in range(h)]
    for ind in range(width):
        Matrix[letter_dictionary[sequence[ind]]][ind] = 1

    return np.array(Matrix)


def read_next_batch(similarity_graph, fasta_file, batchSize):
    time1 = timeit.default_timer()
    reads_dict = read_fasta_file_as_dict(fasta_file)
    time2 = timeit.default_timer()

    # Converts to the right format
    one_hot_encoded_dict = {}
    for key in reads_dict:
        one_hot_encoded_dict[key] = binarize_sequence2(reads_dict[key].seq)

    with open(similarity_graph, 'r') as csvfile:
        pairs = list(csv.reader(csvfile, delimiter=','))

        number_of_pairs = len(pairs)
        number_of_batches = number_of_pairs // batchSize

        for i in range(number_of_batches):
            current_batch = []
            start = batchSize * i
            end = start + batchSize

            for ind in range(start, end):
                current_pair = pairs[ind]
                seq1 = one_hot_encoded_dict[current_pair[0]]
                seq2 = one_hot_encoded_dict[current_pair[1]]
                current_batch.append(np.concatenate((seq1, seq2)))

            yield np.array(current_batch), number_of_batches, batchSize

        # Last batch
        current_batch = []
        start = number_of_batches * batchSize
        end = number_of_pairs
        batchSize = end - start

        for ind in range(start, end):
            current_pair = pairs[ind]
            seq1 = one_hot_encoded_dict[current_pair[0]]
            seq2 = one_hot_encoded_dict[current_pair[1]]
            current_batch.append(np.concatenate((seq1, seq2)))

        yield np.array(current_batch), number_of_batches, batchSize


tf.reset_default_graph()


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 down samples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_1x4(x):
    """max_pool_1x24downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1], padding='SAME')


# Create some variables.
W_conv1 = tf.get_variable("W1", shape=[8, 8, 1, 32])
b_conv1 = tf.get_variable("b1", shape=[32])

W_conv2 = tf.get_variable("W2", shape=[5, 5, 32, 64])
b_conv2 = tf.get_variable("b2", shape=[64])

W_fc1 = tf.get_variable("W3", shape=[4 * 50 * 64, 50])
b_fc1 = tf.get_variable("b3", shape=[50])

W_fc2 = tf.get_variable("W4", shape=[50, 1])
b_fc2 = tf.get_variable("b4", shape=[1])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "../model/CNNModel.ckpt")
    print("Model restored.")
    # Check the values of the variables

    t1 = timeit.default_timer()
    data_generator = read_next_batch('G4.csv', 'trimmed.fasta', 5000)

    label_array = []

    features, batch_num, batch_size = next(data_generator)
    t_end = timeit.default_timer()
    print("One hot encode time: ", t_end - t1)

    print(features.shape)
    first = last = 0
    for i in range(batch_num + 1):
        if i % 10 == 0:
            first = timeit.default_timer()

        r1 = tf.placeholder(tf.float32, [None, 3200])

        features = np.reshape(features, [batch_size, 3200])

        reshaped_data = tf.reshape(r1, [-1, 8, 400, 1])

        h_conv1 = tf.nn.relu(conv2d(reshaped_data, W_conv1) + b_conv1)

        h_pool1 = max_pool_1x4(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        h_pool2 = max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 50 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        final_pred = tf.cast(tf.sign(y_conv), tf.float32)

        predictions = final_pred.eval(feed_dict={r1: features})

        for j in range(len(predictions)):
            label_array.append(predictions[j][0])

        if i == batch_num:
            break

        # test
        # correct_prediction = tf.equal(final_pred, labels)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        features, batch_num, batch_size = next(data_generator)

        if i % 10 == 9:
            last = timeit.default_timer()
            print(last - first)

    with open("Net4.csv", 'w', newline='') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f, delimiter=',')
        for item in label_array:
            w.writerow([item])

    t2 = timeit.default_timer()
    print(t2 - t1)
