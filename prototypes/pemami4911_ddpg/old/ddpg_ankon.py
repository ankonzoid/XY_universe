"""

 ddpg_ankon.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import tensorflow as tf

def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope=scope)

def dense_relu(x, size, scope):
    with tf.variable_scope(scope):
        h1 = dense(x, size, 'dense')
        return tf.nn.relu(h1, 'relu')

def dense_batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, 100, activation_fn=None, scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn')
        return tf.nn.relu(h2, 'relu')

def dense_relu_batch(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = dense_relu(x, 100, scope='dense_relu')
        bn = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn')
        return bn

class Agent:

    def __init__(self):
        self._build_model()

    def _build_model(self):
        tf.reset_default_graph()

        # Build placeholders for input/output
        X = tf.placeholder('float32', (None, 784), name='x')
        y = tf.placeholder('float32', (None, 10), name='y')
        phase = tf.placeholder(tf.bool, name='phase')

        # Build network architecture
        h1 = dense_batch_relu(X, phase, 'layer1')
        h2 = dense_batch_relu(h1, phase, 'layer2')
        logits = dense(h2, 10, 'logits')

        # Define accuracy
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), 'float32'))

        # Define loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    def train(self, X_train, y_train, X_test, y_test):

        accuracy, loss = self.accuracy, self.loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        history = []
        n_epochs = 30
        batch_size = 100
        n_batches = int(X_train.shape[0] / batch_size)
        for epoch in range(n_epochs):
            for i in range(n_batches):
                X_train_batch = X_train[i*batch_size:(i+1)*batch_size]
                y_train_batch = y_train[i*batch_size:(i+1)*batch_size]
                sess.run(train_op, feed_dict={'x:0': X_train_batch, 'y:0': y_train_batch, 'phase:0': 1})  # train
                if i == n_batches-1:
                    loss_train, acc_train = sess.run([loss, accuracy],
                                                     feed_dict={'x:0': X_train, 'y:0': y_train, 'phase:0': 1})  # eval
                    loss_test, acc_test = sess.run([loss, accuracy],
                                                   feed_dict={'x:0': X_test, 'y:0': y_test, 'phase:0': 0})  # eval
                    history += [[epoch] + [loss_train, acc_train] + [loss_test, acc_test]]
                    print("[epoch={}] acc_train={:.3f}, acc_test={:.3f} (L_train={:.5f}, L_test={:.5f}) ".format(
                        epoch, acc_train, acc_test, loss_train, loss_test))
        history = np.array(history)
        return history


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    agent = Agent()
    agent.train(X_train, y_train, X_test, y_test)
