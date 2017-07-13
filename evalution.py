import datetime

import os
import numpy as np
import tensorflow as tf
import image_reader
import model

# PATH_TRAIN = "image"
PATH_EVAL = "data/test"
PATH_SUMMARY = "log/summary"
PATH_MODEL = "log/model_dog&cat"

BATCH_SIZE = 100

if __name__ == '__main__':
    test_batch_images, test_batch_labels = image_reader.get_eval_batch(PATH_EVAL, BATCH_SIZE)

    logits = model.inference(test_batch_images, 4, 1)
    accuracy = model.accuracy(logits, test_batch_labels)

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
            print("Start eval...")
            step_count = int(1000 / BATCH_SIZE)
            accuracy_total = []
            for step in range(step_count):
                accuracy_value = accuracy.eval()
                print("====  %f" % accuracy_value)
                accuracy_total.append(accuracy_value)
            accuracy_final = np.mean(accuracy_total)
            print("The accuracy is %f" % accuracy_final)
        except tf.errors.OutOfRangeError as e:
            print("Error %s" % str(e))
        finally:
            coord.request_stop()
        coord.join(threads)

