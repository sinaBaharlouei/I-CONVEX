import tensorflow as tf
from BNN.buildModel import cnn_model_fn
from BNN.readData import get_train_and_validation_tensors

# Create the Estimator
classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/my_bnn")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)

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
