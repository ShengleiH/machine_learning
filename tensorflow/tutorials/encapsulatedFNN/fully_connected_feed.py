from __future__ import division

import os.path
import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import naiveFNN

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
FLAGS = flags.FLAGS


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, naiveFNN.IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed
    }
    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0;
    '''
    '//': return a integer that is smaller than the real result
    '/': return real result
    '''
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    '''
    xrange performs better than range when a large array of number is produced
    >>> range(5)
    [0, 1, 2, 3, 4]

    >>> xrange(5)
    xrange(5)
    >>> list(xrange(5))
    [0, 1, 2, 3, 4]
    '''
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict)

    precision = true_count / num_examples
    print(' Num examples: %d Num correct: %d Precision @ 1: %0.04f' %(num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
    '''
    It doesn't matter what the value of train_dir is. Because the real url has been defined in read_data_sets function,
    thus the downloading path depends on that url instead of the value, which is the work_directory
    '''

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        logits = naiveFNN.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        loss = naiveFNN.loss(logits, labels_placeholder)
        train_op = naiveFNN.training(loss, FLAGS.learning_rate)

        eval_correct = naiveFNN.evaluation(logits, labels_placeholder)

        summary = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        sess.run(init)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

                print('Training Data Eval: ')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)

                print('Validation Data Eval: ')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)

                print('Test Data Eval: ')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
