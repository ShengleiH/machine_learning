import tensorflow as tf
import cifar10_input
import os

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_CLASSES = cifar10_input.NUM_CLASSES

NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch.')
flags.DEFINE_string('data_dir', 'cifar10_data', 'Path to the CIFAR-10 data directory.')
flags.DEFINE_boolean('use_fp16', False, 'Train the model using fp16.')
FLAGS = flags.FLAGS


def distorted_inputs():
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)
    return var


def inference(images_after_reshape):
    # convolution layer 1
    with tf.variable_scope('conv1') as conv1_scope:
        kernel = _variable_with_weight_decay('weights', shape=[5,5,3,64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(images_after_reshape, kernel, strides=[1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, bias=biases)
        conv1 = tf.nn.relu(conv, name=conv1_scope.name)

    # pooling layer
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

    # normalization
    norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # convolution layer 2
    with tf.variable_scope('conv2') as conv2_scope:
        kernel = _variable_with_weight_decay('weights', shape=[5,5,64,64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases', shape=[64], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.bias_add(conv, bias=biases)
        conv2 = tf.nn.relu(conv, name=conv2_scope.name)

    # normalization
    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

    # pooling layer
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')

    # fully connected layer 1
    with tf.variable_scope('local1') as local1_scope:
        images_as_long_vector = tf.reshape(pool2, shape=[FLAGS.batch_size, -1])
        input_neurons = images_as_long_vector.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[input_neurons, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', shape=[384], initializer=tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(images_as_long_vector, weights) + biases, name=local1_scope.name)

    # fully connected layer 2
    with tf.variable_scope('local2') as local2_scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', shape=[192], initializer=tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(local1, weights) + biases, name=local2_scope.name)

    # output layer - softmax layer
    with tf.variable_scope('softmax_linear') as softmax_scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local1, weights), biases, name=softmax_scope.name)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss


def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

    # Begin training
    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op
