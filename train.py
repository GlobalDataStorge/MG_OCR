import datetime

import os
import numpy as np
import tensorflow as tf
import image_reader
import model

# PATH_TRAIN = "image"
PATH_TRAIN = "data/train_"
PATH_SUMMARY = "log/summary"
PATH_MODEL = "log/model_dog&cat"

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 100
ITERATOR = 10  # step = ITERATOR * TOTAL_IMAGE_COUNT / BATCH_SIZE
TOTAL_IMAGE_COUNT = 1e5

if __name__ == '__main__':
    train_batch_images, train_batch_labels = image_reader.get_train_batch(
        PATH_TRAIN, IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE)

    logits = model.inference(train_batch_images, 4, 0.75)
    loss = model.loss(logits, train_batch_labels)
    # accuracy = model.accuracy(model.inference(test_batch_images, 4), test_batch_labels)
    accuracy = model.accuracy(logits, train_batch_labels)
    train = model.train(loss, LEARNING_RATE)

    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(PATH_SUMMARY, tf.get_default_graph())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(PATH_MODEL)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Load last model successfully.")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            print("Start train...")
            step_count = int(ITERATOR * TOTAL_IMAGE_COUNT / BATCH_SIZE)
            for step in range(1, step_count + 1):
                train.run()
                if step % 10 == 0 or step == 1:
                    prediction = tf.argmax(logits, 1)
                    loss_value, accuracy_value = sess.run([loss, accuracy])
                    # print(train_batch_labels.eval())
                    # print(prediction.eval())
                    # print((train_batch_images[0:5]).eval())
                    print("Time %s, Step %d, Loss %f Accuracy %f" % (datetime.datetime.now(), step, loss_value, accuracy_value))
                if step % 30 == 0:
                    summary_result = sess.run(summary)
                    summary_writer.add_summary(summary_result, step)
                if step % 50 == 0 or step == step_count:
                    saver.save(sess, os.path.join(PATH_MODEL, "model"), step)
        except tf.errors.OutOfRangeError as e:
            print("Error %s" % str(e))
        finally:
            coord.request_stop()
        coord.join(threads)
