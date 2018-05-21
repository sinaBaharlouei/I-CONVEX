import csv
import tensorflow as tf
from BNN import readData
import numpy as np
import timeit
from DataOperations import FastaIO

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

    data_generator = FastaIO.read_next_batch('MinGraphK15R1B10P10.csv', '../files/reads400.fasta', 5000)

    label_array = []

    features, batch_num, batch_size = next(data_generator)
    print(features.shape)
    for i in range(batch_num+1):
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
        last = timeit.default_timer()
        print(last - first)

    with open("MHNET10_400.csv", 'w', newline='') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f, delimiter=',')
        for item in label_array:
            w.writerow([item])
