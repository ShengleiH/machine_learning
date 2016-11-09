import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def init_weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Prepare placeholders and initialize variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W1 = init_weights([784, 100])
b1 = init_bias([100])
W2 = init_weights([100, 10])
b2 = init_bias([10])

# Predict labels
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

# Back Propagation and Train model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Evaluate model
correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run from NOW...
session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Test model
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
