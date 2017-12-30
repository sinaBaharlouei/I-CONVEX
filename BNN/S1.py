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

from BNN import readData
import tensorflow as tf
import numpy as np

FLAGS = None


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 8, 400, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([8, 8, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        print(h_conv1.shape)

    # Pooling layer - downsamples by 4X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_1x4(h_conv1)
        print(h_pool1.shape)
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        print(h_pool2.shape)

    # Fully connected layer 1 -- after 2 round of down sampling, our 28x28 image
    # is down to 4x50x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 50 * 64, 50])
        b_fc1 = bias_variable([50])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 50 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([50, 1])
        b_fc2 = bias_variable([1])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2


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


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


data = readData.read_data_sets("../files/pair_dataset_1000.csv", 0.1, 0.1)
# Create the model
r1 = tf.placeholder(tf.float32, [None, 3200])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
y_conv, keep_prob, W1, b1, W2, b2, W3, b3, W4, b4 = deepnn(r1)

# training
cross_entropy = tf.reduce_mean(tf.square(y_ - y_conv), keep_dims=True)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# test
final_pred = tf.cast(tf.sign(y_conv), tf.float32)
correct_prediction = tf.equal(final_pred, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver({'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4,})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # training
    for i in range(100):
        features, labels = data.train.next_batch(100)

        features = np.reshape(features, [100, 3200])
        labels = np.reshape(labels, [100, 1])

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={r1: features, y_: labels, keep_prob: 1})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            # print(y_conv.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
            # print(final_pred.eval(feed_dict={r1: features, y_: labels, keep_prob: 0.5}))

        train_step.run(feed_dict={r1: features, y_: labels, keep_prob: 0.4})
    batch_size = 100
    batch_num = int(data.test.num_examples / batch_size)
    test_accuracy = 0
    for i in range(batch_num):
        features, labels = data.test.next_batch(batch_size)

        labels = np.reshape(labels, [100, 1])
        features = np.reshape(features, [100, 3200])
        print('test accuracy %g' % accuracy.eval(feed_dict={r1: features, y_: labels, keep_prob: 1}))
        # print(final_pred.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
        # print(y_conv.eval(feed_dict={r1: x1, r2: x2, y_: labels}))
    test_accuracy /= batch_num
    print("test accuracy %g" % test_accuracy)

    save_path = saver.save(sess, "../model/CNNModel.ckpt")
    print("Model saved in file: %s" % save_path)
