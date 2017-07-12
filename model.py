import tensorflow as tf


def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def biases_variables(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W, b, strides, activation=tf.nn.relu):
    result = tf.add(tf.nn.conv2d(x, W, strides, padding="SAME"), b)
    if activation is not None:
        result = activation(result)
    return result


def max_pool(x, ksize, strides):
    return tf.nn.max_pool(x, ksize, strides, padding="SAME")


def inference(images):
    images_stream = images
    conv_layer = [32, 64]
    current_layer = images[3]
    for next_conv_layer in conv_layer:
        W_conv = weight_variables([3, 3, current_layer, next_conv_layer])
        b_conv = biases_variables([next_conv_layer])
        h_conv = conv2d(images, W_conv, b_conv, [1, 1, 1, 1])
        images_stream = max_pool(h_conv, [1, 2, 2, 1], [1, 2, 2, 1])
        current_layer = next_conv_layer

    images_stream = tf.reshape(images_stream, [images[0], -1])
    current_input_num = images_stream.get_shape()[-1]
    fc_layer = [256, 512, 512]
    for i, next_fc_layer in enumerate(fc_layer):
        W_fc = weight_variables([current_input_num, next_fc_layer])
        b_fc = biases_variables([next_fc_layer])
        images_stream = tf.add(tf.matmul(images_stream, W_fc), b_fc)
        if i == len(fc_layer) - 1:
            images_stream = tf.nn.softmax(images_stream)
        else:
            images_stream = tf.nn.relu(images_stream)

    return images_stream


def accuracy(logits, label):
    correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)
    accuracy_value = tf.reduce_mean(correct)
    tf.summary.scalar("accuracy", accuracy_value)
    return accuracy_value


def loss(logits, labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def train(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss)
