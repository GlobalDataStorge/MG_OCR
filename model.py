import tensorflow as tf


def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def biases_variables(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


def conv2d(x, W, b, strides, activation=tf.nn.relu):
    result = tf.add(tf.nn.conv2d(x, W, strides, padding="SAME"), b)
    if activation is not None:
        result = activation(result)
    return result


def max_pool(x, ksize, strides):
    return tf.nn.max_pool(x, ksize, strides, padding="SAME")


def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, is_training=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    epsilon = 1e-3
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)


def inference(images, n_classes, keep_prob):
    tf.summary.image("images", images)
    images_stream = images
    conv_layer = [64, 128, 256, 256]
    current_layer = int(images.get_shape()[3])
    for i, next_conv_layer in enumerate(conv_layer):
        W_conv = weight_variables([5, 5, current_layer, next_conv_layer])
        b_conv = biases_variables([next_conv_layer])
        h_conv = conv2d(images_stream, W_conv, b_conv, [1, 2, 2, 1])
        images_stream = max_pool(h_conv, [1, 2, 2, 1], [1, 2, 2, 1])
        # images_stream = tf.nn.lrn(images_stream, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
        #                           beta=0.75)
        current_layer = next_conv_layer

    batch_size = int(images.get_shape()[0])
    images_stream = tf.reshape(images_stream, [batch_size, -1])
    current_input_num = int(images_stream.get_shape()[-1])
    fc_layer = [512, 1024, n_classes]
    for i, next_fc_layer in enumerate(fc_layer):
        W_fc = weight_variables([current_input_num, next_fc_layer])
        current_input_num = next_fc_layer
        b_fc = biases_variables([next_fc_layer])
        images_stream = tf.add(tf.matmul(images_stream, W_fc), b_fc)
        if i != len(fc_layer) - 1:
            images_stream = tf.nn.relu(images_stream)
            # images_stream = tf.nn.dropout(images_stream, keep_prob)
            # images_stream = batch_norm(images_stream)
            images_stream = batch_norm_wrapper(images_stream)
    return images_stream


def accuracy(logits, label):
    correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32)
    accuracy_value = tf.reduce_mean(correct)
    tf.summary.scalar("accuracy", accuracy_value)
    return accuracy_value


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    tf.summary.scalar("loss", loss)
    return loss


def train(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss)
