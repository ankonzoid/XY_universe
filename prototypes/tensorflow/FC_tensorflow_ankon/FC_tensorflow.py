"""

 FC_tensorflow.py  (author: Anson Wong / git: ankonzoid)

"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, 28, 28))
y_test = mnist.test.labels

# Build model, loss, training, evaluation
n_input = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("arch"):
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
    bn_params = {"is_training": is_training, "decay": 0.99, "update_collections": None}
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1", normalizer_fn=batch_norm, normalizer_params=bn_params)
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", normalizer_fn=batch_norm, normalizer_params=bn_params)
    output = fully_connected(hidden2, n_output, scope="outputs", normalizer_fn=batch_norm, normalizer_params=bn_params)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(output, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train model
init = tf.global_variables_initializer()
n_epochs = 100
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        acc_train = 0.0
        n_batches = mnist.train.num_examples // batch_size
        for iter in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train += accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_train /= n_batches
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("[epoch {}] acc_train={}, acc_test={}".format(epoch, acc_train, acc_test))
