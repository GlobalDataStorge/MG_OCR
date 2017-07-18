import os
import numpy as np
import tensorflow as tf

import model

PATH_MODEL = "log/model_dog&cat"


def parse(image_files, image_width=100, image_height=100, channels=3):
    images = get_images(channels, image_files, image_height, image_width)
    logits = model.inference(images, 4, 1)
    classify = tf.nn.softmax(logits)
    # classify = logits
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(PATH_MODEL)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print "Load model successfully."
        classify_value = sess.run(classify)
        prediction = [list(item) for item in classify_value]
        return prediction
        # return np.argmax(prediction, 1)
        # return tf.argmax(classify, 1).eval()


def get_images(channels, image_files, image_height, image_width):
    images = []
    for image_file in image_files:
        if not os.path.exists(image_file):
            raise IOError("File %s not found." % image_file)
        image_content = tf.read_file(image_file)
        image = tf.image.decode_jpeg(image_content, channels)
        image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
        image = tf.image.per_image_standardization(image)
        image = tf.transpose(image, (1, 0, 2))
        image = tf.expand_dims(image, 0)
        images.append(image)
    images = tf.concat(images, 0)
    return images


if __name__ == '__main__':
    # image_files = ["data/test/%d_1.jpg" % (i + 10000) for i in range(1, 100)]

    image_files = []
    for i in range(4):
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


