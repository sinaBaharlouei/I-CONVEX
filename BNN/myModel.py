import argparse
import sys
from BNN import readData
import tensorflow as tf
import numpy as np


def deepnn(x1, x2):

    read1 = tf.reshape(x1, [-1, 4, 400, 1])
    read2 = tf.reshape(x2, [-1, 4, 400, 1])

    print("Mine: ", read1.shape)
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    """Computes a 2-D convolution given 4-D `input` and `filter` tensors.

      Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
      and a filter / kernel tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`, this op
      performs the following:
    """

    W_conv1 = weight_variable([4, 8, 1, 32])
    b_conv1 = bias_variable([32])

    h1_conv1 = tf.nn.relu(conv2d(read1, W_conv1) + b_conv1)
    h2_conv1 = tf.nn.relu(conv2d(read2, W_conv1) + b_conv1)
    print(h1_conv1.shape)
    # output: (100, 1, 100, 32)

    # Pooling layer - downsamples by 2X.
    h1_pool1 = max_pool_1x2(h1_conv1)
    h2_pool1 = max_pool_1x2(h2_conv1)

    print(h1_pool1.shape)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([1, 4, 32, 20])
    b_conv2 = bias_variable([20])
    h1_conv2 = tf.nn.relu(conv2d(h1_pool1, W_conv2) + b_conv2)
    h2_conv2 = tf.nn.relu(conv2d(h2_pool1, W_conv2) + b_conv2)

    print("conv2: ", h2_conv2.shape)
    # Second pooling layer.
    h1_pool2 = max_pool_1x2(h1_conv2)
    h2_pool2 = max_pool_1x2(h2_conv2)

    print("pool2: ", h2_pool2.shape)

    W_fc1 = weight_variable([500, 30])
    b_fc1 = bias_variable([30])
    b2_fc1 = bias_variable([30])

    h1_pool2_flat = tf.reshape(h1_pool2, [-1, 500])
    h2_pool2_flat = tf.reshape(h2_pool2, [-1, 500])
    print("flattened: ", h2_pool2_flat.shape)

    tanh_beta = tf.constant(1.0)
    tanh_beta2 = tf.constant(1.0)

    h1_fc1 = tf.nn.tanh(tf.multiply(tanh_beta2, tf.nn.relu(tf.matmul(h1_pool2_flat, W_fc1))) + b2_fc1)
    print("fully connected output: ", h1_fc1.shape)

    h2_fc1 = tf.nn.tanh(tf.multiply(tanh_beta2, tf.nn.relu(tf.matmul(h2_pool2_flat, W_fc1))) + b2_fc1)

    b_final = bias_variable([1])

    res = tf.multiply(h1_fc1, h2_fc1) + b_final

    inner_product = tf.reduce_sum(res, 1)
    print("Inner product shape: ", inner_product.shape)
    non_linearity = tf.nn.tanh(
        tf.multiply(tanh_beta, inner_product))

    print(non_linearity.shape)
    return non_linearity, inner_product, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1
    # return inner_product


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')


def conv2d_1(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 2, 1], padding='SAME')


def max_pool_1x2(x):
    """max_pool_2x1 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


data = readData.read_data_sets("../files/pairs_100.csv", 0.1, 0.1)
# Create the model
r1 = tf.placeholder(tf.float32, [100, 4, 400, 1])
r2 = tf.placeholder(tf.float32, [100, 4, 400, 1])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [100, 1])

# Build the graph for the deep net
y_conv, inner_product, W1_temp, B1_temp, W2_temp, B2_temp, W3_temp, B3_temp = deepnn(r1, r2)

# training
cross_entropy = tf.reduce_mean(tf.square(y_ - y_conv), keep_dims=True)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# test
final_pred = tf.cast(tf.sign(y_conv), tf.float32)
correct_prediction = tf.equal(final_pred, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # training
    for i in range(1000):
        features, labels = data.train.next_batch(100)

        parts = np.split(features, 2, axis=1)
        x1 = parts[0]
        x2 = parts[1]
        labels = np.reshape(labels, [100, 1])

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={r1: x1, r2: x2, y_: labels})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            print(final_pred.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
        train_step.run(feed_dict={r1: x1, r2: x2, y_: labels})

    batch_size = 100
    batch_num = int(data.test.num_examples / batch_size)
    test_accuracy = 0
    for i in range(batch_num):
        features, labels = data.test.next_batch(batch_size)
        parts = np.split(features, 2, axis=1)
        x1 = parts[0]
        x2 = parts[1]
        labels = np.reshape(labels, [100, 1])

        accuracy.eval(feed_dict={r1: x1, r2: x2, y_: labels})
        test_accuracy += accuracy.eval(feed_dict={r1: x1, r2: x2, y_: labels})
        print(final_pred.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
    test_accuracy /= batch_num
    print("test accuracy %g" % test_accuracy)


"""

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 5000])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print(batch_xs.shape)
        print(batch_ys.shape)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main)
"""
print("hi")
