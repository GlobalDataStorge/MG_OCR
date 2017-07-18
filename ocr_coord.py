import random

import os
import numpy as np
import tensorflow as tf

import model

PATH_MODEL = "log/model_dog&cat"


def read_image(image_files):
    queue = tf.train.slice_input_producer([image_files], shuffle=False)
    image_data = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_data, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, 100, 100)
    image = tf.image.resize_images(image, (100, 100))
    image = tf.image.per_image_standardization(image)
    images = tf.train.batch([image], len(image_files))
    images = tf.cast(images, tf.float32)
    return images


def parse(image_files):
    images = read_image(image_files)
    logits = model.inference(images, 4, 1)
    classify = tf.nn.softmax(logits)
    with tf.Session() as sess:
        prediction = []
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(PATH_MODEL)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print "Load model successfully."
            classify_value = sess.run(classify)
            prediction = [list(item) for item in classify_value]
        except tf.errors.OutOfRangeError, e:
            print e
        finally:
            coord.request_stop()
        coord.join(threads)
        return prediction


if __name__ == '__main__':
    # image_files = ["data/test/%d_%d.jpg" % (i + 10201, random.randint(0, 3)) for i in range(200)]

    image_files = []
    for i in range(100):
        for j in range(4):
            image_files.append("data/test/%d_%d.jpg" % (i + 10001, j))

    results = parse(image_files)
    prediction = []
    for i in range(len(image_files)):
        file_name = image_files[i].split("/")[-1]
        result = np.argmax(results[i])
        print file_name, results[i]
        true = file_name.split(".")[0].endswith(str(result))
        prediction.append(int(true))
    print "Accuracy is %f" % np.mean(prediction)
