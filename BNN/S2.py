# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
from BNN import readData
import tensorflow as tf
import numpy as np
import csv

FLAGS = None


def deepnn(x1, x2):
    """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
    :param x2:
    :param x1:
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        pair1 = tf.reshape(x1, [-1, 4, 400, 1])
        pair2 = tf.reshape(x2, [-1, 4, 400, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([4, 4, 1, 32])
        b_conv1 = bias_variable([32])

        h1_conv1 = tf.nn.relu(conv2d(pair1, W_conv1) + b_conv1)
        h2_conv1 = tf.nn.relu(conv2d(pair2, W_conv1) + b_conv1)

        print(h1_conv1.shape)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h1_pool1 = max_pool_1x4(h1_conv1)
        h2_pool1 = max_pool_1x4(h2_conv1)
        print(h1_pool1.shape)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([2, 5, 32, 20])
        b_conv2 = bias_variable([20])

        h1_conv2 = tf.nn.relu(conv2d(h1_pool1, W_conv2) + b_conv2)
        h2_conv2 = tf.nn.relu(conv2d(h2_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h1_pool2 = max_pool_2x2(h1_conv2)
        h2_pool2 = max_pool_2x2(h2_conv2)

        print(h1_pool2.shape)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([2 * 50 * 20, 50])
        b2_fc1 = bias_variable([50])

        h1_pool2_flat = tf.reshape(h1_pool2, [-1, 2 * 50 * 20])
        h2_pool2_flat = tf.reshape(h2_pool2, [-1, 2 * 50 * 20])

        tanh_beta = tf.constant(1.0)
        tanh_beta2 = tf.constant(1.0)

        h1_fc1 = tf.nn.tanh(tf.matmul(h1_pool2_flat, W_fc1) + b2_fc1)
        h2_fc1 = tf.nn.tanh(tf.matmul(h2_pool2_flat, W_fc1) + b2_fc1)

        print(h1_fc1.shape)

        h1_square = tf.square(h1_fc1)
        h2_square = tf.square(h2_fc1)

        ones = tf.ones([100, 50])

        h1_reg = tf.subtract(h1_square, ones)
        h2_reg = tf.subtract(h2_square, ones)

        h1_reg_square = tf.square(h1_reg)
        h2_reg_square = tf.square(h2_reg)

        final_reg1 = tf.reduce_sum(h1_reg_square, 1, keep_dims=True)
        final_reg2 = tf.reduce_sum(h2_reg_square, 1, keep_dims=True)

        print("final reg shape: ", final_reg2.shape)

        normalized1 = tf.nn.l2_normalize(h1_fc1, dim=1)
        normalized2 = tf.nn.l2_normalize(h2_fc1, dim=1)

        inner_product = tf.reduce_sum(tf.multiply(normalized1, normalized2), 1, keep_dims=True)
        print(inner_product.shape)

        non_linearity = tf.tanh(inner_product)

        return non_linearity, h1_fc1, h2_fc1, h1_reg_square, h2_reg_square, final_reg1, final_reg2


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_1x4(x):
    """max_pool_1x24downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                          strides=[1, 1, 4, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


data = readData.read_data_sets("../files/pair_dataset_20000.csv", 0.1, 0.1)
# Create the model
r1 = tf.placeholder(tf.float32, [None, 1600])
r2 = tf.placeholder(tf.float32, [None, 1600])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
y_conv, hash1, hash2, reg1, reg2, finalReg1, finalReg2 = deepnn(r1, r2)

# training
cross_entropy = tf.reduce_mean(tf.square(y_ - y_conv), keep_dims=True)
# cross_entropy = tf.reduce_mean(tf.add(tf.scalar_mul(500, tf.square(y_ - y_conv)), tf.add(finalReg1, finalReg2)), keep_dims=True)

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# test
final_pred = tf.cast(tf.sign(y_conv), tf.float32)
correct_prediction = tf.equal(final_pred, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # training
    for i in range(5000):
        features, labels = data.train.next_batch(100)

        parts = np.split(features, 2, axis=1)

        feature1 = np.reshape(parts[0], [100, 1600])
        feature2 = np.reshape(parts[1], [100, 1600])

        labels = np.reshape(labels, [100, 1])

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={r1: feature1, r2: feature2, y_: labels})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            # print(y_conv.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
            # print(final_pred.eval(feed_dict={r1: feature1, r2: feature2, y_: labels}))

        train_step.run(feed_dict={r1: feature1, r2: feature2, y_: labels})

    batch_size = 100
    batch_num = int(data.test.num_examples / batch_size)
    test_accuracy = 0

    with open("hashes2.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in range(batch_num):
            features, labels = data.test.next_batch(batch_size)

            labels = np.reshape(labels, [100, 1])

            parts = np.split(features, 2, axis=1)

            feature1 = np.reshape(parts[0], [100, 1600])
            feature2 = np.reshape(parts[1], [100, 1600])

            print('test accuracy %g' % accuracy.eval(feed_dict={r1: feature1, r2: feature2, y_: labels}))
            # print(final_pred.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
            # print(y_conv.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
            list1 = list(hash1.eval(feed_dict={r1: feature1, r2: feature2, y_: labels}))
            list2 = list(hash2.eval(feed_dict={r1: feature1, r2: feature2, y_: labels}))

            for item in list1:
                for element in item:
                    writer.writerow([element])

            for item in list2:
                for element in item:
                    writer.writerow([element])

                # print(list(hash2.eval(feed_dict={r1: feature1, r2: feature2, y_: labels}))[0])
        # print(list(reg1.eval(feed_dict={r1: feature1, r2: feature2, y_: labels}))[0])

    test_accuracy /= batch_num
    print("test accuracy %g" % test_accuracy)
