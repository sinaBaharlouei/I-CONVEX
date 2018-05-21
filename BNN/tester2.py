import tensorflow as tf
from BNN import readData
import numpy as np
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
W_conv1 = tf.get_variable("W1", shape=[4, 4, 1, 32])
b_conv1 = tf.get_variable("b1", shape=[32])

W_conv2 = tf.get_variable("W2", shape=[2, 5, 32, 32])
b_conv2 = tf.get_variable("b2", shape=[32])

W_fc1 = tf.get_variable("W3", shape=[2 * 50 * 32, 50])
b_fc1 = tf.get_variable("b3", shape=[50])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "../model/regularized_hashnet.ckpt")
    print("Model restored.")
    # Check the values of the variables

    data = readData.read_data_sets("../files/pair_dataset_100000.csv", 0.1, 0.1)

    features, labels = data.test.next_batch(100)

    r1 = tf.placeholder(tf.float32, [None, 3200])
    y_ = tf.placeholder(tf.float32, [None, 1])

    labels = np.reshape(labels, [100, 1])
    features = np.reshape(features, [100, 3200])

    reshaped_data = tf.reshape(r1, [-1, 8, 400, 1])

    h_conv1 = tf.nn.relu(conv2d(reshaped_data, W_conv1) + b_conv1)

    h_pool1 = max_pool_1x4(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 50 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    final_pred = tf.cast(tf.sign(y_conv), tf.float32)

    # test
    correct_prediction = tf.equal(final_pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval(feed_dict={r1: features, y_: labels}))