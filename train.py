import datetime
import tensorflow as tf
import image_reader
import model

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 100
MAX_ITERATOR = 1e4

if __name__ == '__main__':
    train_batch_images, train_batch_labels = image_reader.get_train_batch()
    test_batch_images, test_batch_labels = image_reader.get_test_batch()

    logits = model.inference(train_batch_images)
    loss = model.loss(logits, train_batch_labels)
    accuracy = model.accuracy(model.inference(test_batch_images), test_batch_images)
    train = model.train(loss, LEARNING_RATE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for step in range(1, MAX_ITERATOR + 1):
                train.run()
                if step % 50 == 0:
                    loss_value, accuracy_value = sess.run([loss, accuracy])
                    print "Time %s, Step %d, Loss %f Accuracy %f" % (
                        datetime.datetime.now(), step, loss_value, accuracy_value)
        except tf.errors.OutOfRangeError:
            print "Error"
        finally:
            coord.request_stop()
        coord.join(threads)
