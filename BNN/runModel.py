import tensorflow as tf
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


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 8, 400, 1])
    input_layer = tf.cast(input_layer, tf.float16)

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[8, 9],
        padding="valid",
        activation=tf.nn.relu)

    print(conv1.shape)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 4], strides=4)
    print(pool1.shape)

    pool2_flat = tf.reshape(pool1, [-1, 1 * 193 * 32])

    print(pool2_flat.shape)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Create the Estimator
classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/my_bnn")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=10)

train_data, train_labels, eval_data, eval_labels = get_train_and_validation_tensors('../pairs_100.csv', 0.8)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
classifier.train(
    input_fn=train_input_fn,
    steps=10000,
    hooks=[logging_hook])

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
