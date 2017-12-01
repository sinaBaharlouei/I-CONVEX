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

# from tensorflow.examples.tutorials.mnist import input_data

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

import tensorflow as tf


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=False):

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
      ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    #  with gzip.GzipFile(fileobj=f) as bytestream:
    #  with gzip.GzipFile(fileobj=f) as bytestream:
    bytestream = f
    magic = _read32(bytestream)
    print('Extracted %d' % magic)
    if magic != 2051:
        raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                         (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    print(num_images, rows, cols)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    print(num_images, rows, cols)
    data = data.reshape(num_images, rows, cols, 1) * 255.0
    return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=2):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.
    Returns:
      labels: a 1D uint8 numpy array.
    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)

    #  with gzip.GzipFile(fileobj=f) as bytestream:
    bytestream = f
    magic = _read32(bytestream)
    print('Extracted %d' % magic)
    if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                         (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8) * 2.0 - 1.0
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    labels = labels.reshape(num_items, 1)
    return labels


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
    if fake_data:
        def fake():
            return DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = 'train.data.60k.csv.ubyte1'
    TRAIN_LABELS = 'train.label.60k.csv.ubyte'
    TEST_IMAGES = 'test.data.10k.csv.ubyte1'
    TEST_LABELS = 'test.label.10k.csv.ubyte'

    local_file = TRAIN_IMAGES
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = TRAIN_LABELS
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = TEST_IMAGES
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = TEST_LABELS
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)


def read_data_sets2(train_dir,
                    fake_data=False,
                    one_hot=False,
                    dtype=dtypes.float32,
                    reshape=True,
                    validation_size=5000,
                    seed=None):
    if fake_data:
        def fake():
            return DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = 'train.data.60k.csv.ubyte2'
    TRAIN_LABELS = 'train.label.60k.csv.ubyte'
    TEST_IMAGES = 'test.data.10k.csv.ubyte2'
    TEST_LABELS = 'test.label.10k.csv.ubyte'

    local_file = TRAIN_IMAGES
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = TRAIN_LABELS
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = TEST_IMAGES
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = TEST_LABELS
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)


FLAGS = None


def deepnn(x1, x2):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 800), where 800 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 2), with values
      equal to the logits of classifying the digit into one of 2 classes (the
      0:mimsmatch 1:match). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x1_image = tf.reshape(x1, [-1, 100, 4, 1])
    x2_image = tf.reshape(x2, [-1, 100, 4, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([3, 4, 1, 32])
    b_conv1 = bias_variable([32])
    h1_conv1 = tf.nn.relu(conv2d_1(x1_image, W_conv1) + b_conv1)
    h2_conv1 = tf.nn.relu(conv2d_1(x2_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h1_pool1 = max_pool_2x1(h1_conv1)
    h2_pool1 = max_pool_2x1(h2_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 1, 32, 20])
    b_conv2 = bias_variable([20])
    h1_conv2 = tf.nn.relu(conv2d(h1_pool1, W_conv2) + b_conv2)
    h2_conv2 = tf.nn.relu(conv2d(h2_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h1_pool2 = max_pool_2x1(h1_conv2)
    h2_pool2 = max_pool_2x1(h2_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.

    W_fc1 = weight_variable([500, 30])
    b_fc1 = bias_variable([30])
    b2_fc1 = bias_variable([30])

    h1_pool2_flat = tf.reshape(h1_pool2, [-1, 500])
    h2_pool2_flat = tf.reshape(h2_pool2, [-1, 500])

    tanh_beta = tf.constant(1.0)
    tanh_beta2 = tf.constant(1.0)

    h1_fc1 = tf.nn.tanh(tf.multiply(tanh_beta2, tf.nn.relu(tf.matmul(h1_pool2_flat, W_fc1))) + b2_fc1)
    h2_fc1 = tf.nn.tanh(tf.multiply(tanh_beta2, tf.nn.relu(tf.matmul(h2_pool2_flat, W_fc1))) + b2_fc1)

    b_final = bias_variable([1])

    inner_product = tf.nn.tanh(
        tf.multiply(tanh_beta, tf.reduce_sum(tf.multiply(h1_fc1, h2_fc1) + b_final, 1, keep_dims=True)))

    return inner_product, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1
    # return inner_product


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_1(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 4, 1], padding='SAME')


def max_pool_2x1(x):
    """max_pool_2x1 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 1, 1], padding='SAME')


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


def main(_):
    # Import data
    mnist1 = read_data_sets(FLAGS.data_dir, one_hot=False)
    mnist2 = read_data_sets2(FLAGS.data_dir, one_hot=False)

    # Create the model
    x1 = tf.placeholder(tf.float32, [None, 400])
    x2 = tf.placeholder(tf.float32, [None, 400])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Build the graph for the deep net
    y_conv, W1_temp, B1_temp, W2_temp, B2_temp, W3_temp, B3_temp = deepnn(x1, x2)

    # training
    cross_entropy = tf.reduce_mean(tf.square(y_ - y_conv), keep_dims=True)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # test
    final_pred = tf.cast(tf.sign(y_conv), tf.float32)
    correct_prediction = tf.equal(final_pred, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # save the variables
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # training
        for i in range(20000):
            batch1 = mnist1.train.next_batch(50)
            batch2 = mnist2.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x1: batch1[0], x2: batch2[0], y_: batch1[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x1: batch1[0], x2: batch2[0], y_: batch1[1]})

        # save_path = saver.save(sess, "./vars/all_weights")
        # print("Model saved in file: %s" % save_path)

        # print stuff here
        # print(sess.run(W1_temp))
        # print(sess.run(B1_temp))
        # print(sess.run(W2_temp))
        # print(sess.run(B2_temp))
        # print(sess.run(W3_temp))
        # print(sess.run(B3_temp))

        # test
        batch_size = 50
        batch_num = int(mnist1.test.num_examples / batch_size)
        test_accuracy = 0
        for i in range(batch_num):
            batch1 = mnist1.test.next_batch(batch_size)
            batch2 = mnist2.test.next_batch(batch_size)
            test_accuracy += accuracy.eval(feed_dict={x1: batch1[0], x2: batch2[0], y_: batch1[1]})

        test_accuracy /= batch_num
        print("test accuracy %g" % test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
