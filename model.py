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


def inference(images, n_classes, keep_prob):
    tf.summary.image("images", images)
    images_stream = images
    conv_layer = [32, 64]
    current_layer = int(images.get_shape()[3])
    for next_conv_layer in conv_layer:
        W_conv = weight_variables([3, 3, current_layer, next_conv_layer])
        b_conv = biases_variables([next_conv_layer])
        h_conv = conv2d(images_stream, W_conv, b_conv, [1, 1, 1, 1])
        images_stream = max_pool(h_conv, [1, 2, 2, 1], [1, 2, 2, 1])
        current_layer = next_conv_layer

    batch_size = int(images.get_shape()[0])
    images_stream = tf.reshape(images_stream, [batch_size, -1])
    current_input_num = int(images_stream.get_shape()[-1])
    fc_layer = [256, 512, 512, n_classes]
    for i, next_fc_layer in enumerate(fc_layer):
        W_fc = weight_variables([current_input_num, next_fc_layer])
        current_input_num = next_fc_layer
        b_fc = biases_variables([next_fc_layer])
        images_stream = tf.add(tf.matmul(images_stream, W_fc), b_fc)
        if i != len(fc_layer) - 1:
            images_stream = tf.nn.relu(images_stream)
            images_stream = tf.nn.dropout(images_stream, keep_prob)
    return images_stream


def accuracy(logits, label):
    correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)
    accuracy_value = tf.reduce_mean(correct)
    tf.summary.scalar("accuracy", accuracy_value)
    return accuracy_value


def loss(logits, labels):
    cross_entropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    tf.summary.scalar("loss", loss)
    return loss


def train(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss)
