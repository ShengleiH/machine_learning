import tensorflow as tf
import cifar10_model
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = cifar10_model.distorted_inputs()
        logits = cifar10_model.inference(images)
        loss = cifar10_model.loss(logits, labels)
        train_op = cifar10_model.train(loss, global_step)
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            if step % 100 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))


def main(argv=None):
    train()


if __name__ == '__main__':
  tf.app.run()
