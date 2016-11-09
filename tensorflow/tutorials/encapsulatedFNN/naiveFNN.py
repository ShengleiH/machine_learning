import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def weights_varialble(shape, input_units):
    initial = tf.truncated_normal(shape, stddev=1.0/math.sqrt(float(input_units)))
    return tf.Variable(initial, name='weights')


def biases_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name='biases')


def inference(images, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = weights_varialble([IMAGE_PIXELS, hidden1_units], IMAGE_PIXELS)
        biases = biases_variable([hidden1_units])
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = weights_varialble([hidden1_units, hidden2_units], hidden1_units)
        biases = biases_variable([hidden2_units])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = weights_varialble([hidden2_units, NUM_CLASSES], hidden2_units)
        biases = biases_variable([NUM_CLASSES])
        logits = tf.matmul(hidden2, weights) + biases

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))