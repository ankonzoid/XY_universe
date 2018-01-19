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
        agent.build_model()
        pass

    def build_model(self):
        tf.reset_default_graph()

        #  Build placeholders for input/output
        x = tf.placeholder('float32', (None, 784), name='x')
        y = tf.placeholder('float32', (None, 10), name='y')
        phase = tf.placeholder(tf.bool, name='phase')

        # Build network architecture
        h1 = dense_batch_relu(x, phase, 'layer1')
        h2 = dense_batch_relu(h1, phase, 'layer2')
        logits = dense(h2, 10, 'logits')

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)),
                'float32'))

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

        history = train(loss, accuracy)
        history = np.array(history)
        return history


agent = Agent()