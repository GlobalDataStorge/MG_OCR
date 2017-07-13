import tensorflow as tf
import numpy as np
import os


def get_train_batch(path, image_width=100, image_height=100, batch_size=100, num_threads=64, capacity=1000,
                    standardization=True):
    if not os.path.exists(path):
        raise IOError("Files not found.")
    images, labels = [], []
    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            images.append(os.path.join(path, file_name))
            rotate = int(file_name.split(".")[0].split("_")[-1])
            labels.append(rotate)

    # shuffle data
    package = np.array([images, labels])  # order 2 * 40000
    package = package.transpose()  # order 40000 * 2
    np.random.shuffle(package)  # shuffle 4000 * 2
    images = package[:, 0]  # shuffle images 4000 * 1
    labels = package[:, 1]  # shuffle labels 4000 * 1
    labels = [int(i) for i in labels]

    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    queue = tf.train.slice_input_producer([images, labels])
    image_data = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_data, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
    image = tf.image.resize_images(image, (image_height, image_width))
    # image = tf.random_crop(image, [image_height, image_width, 3])
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    if standardization:
        image = tf.image.per_image_standardization(image)
    label = queue[1]
    batch_image, batch_label = tf.train.batch([image, label], batch_size, num_threads, capacity)
    batch_image = tf.cast(batch_image, tf.float32)
    batch_label = tf.cast(batch_label, tf.int32)
    return batch_image, batch_label


def get_eval_batch(path, batch_size):
    if not os.path.exists(path):
        raise IOError("Files not found.")
    images, labels = [], []
    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            images.append(os.path.join(path, file_name))
            rotate = int(file_name.split(".")[0].split("_")[-1])
            labels.append(rotate)

    # shuffle data
    package = np.array([images, labels])  # order 2 * 40000
    package = package.transpose()  # order 40000 * 2
    np.random.shuffle(package)  # shuffle 4000 * 2
    images = package[:, 0]  # shuffle images 4000 * 1
    labels = package[:, 1]  # shuffle labels 4000 * 1
    labels = [int(i) for i in labels]

    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    queue = tf.train.slice_input_producer([images, labels])
    image_data = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize_images(image, (100, 100))
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.per_image_standardization(image)
    label = queue[1]
    batch_image, batch_label = tf.train.batch([image, label], batch_size, 64, 1000)
    batch_image = tf.cast(batch_image, tf.float32)
    batch_label = tf.cast(batch_label, tf.int32)
    return batch_image, batch_label

