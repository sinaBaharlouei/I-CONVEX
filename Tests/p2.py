import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
W_conv1 = tf.get_variable("W1", shape=[8, 8, 1, 32])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "../model/CNNModel.ckpt")
    print("Model restored.")
    # Check the values of the variables
    print("W1 : %s" % W_conv1.eval())
