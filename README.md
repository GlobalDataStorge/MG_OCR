# MG_OCR

*图片方向识别*

#### 模型结构

三个卷积层 + 三个FC层

```
def inference(images, n_classes):
    tf.summary.image("images", images)
    images_stream = images
    conv_layer = [64, 64, 128]
    current_layer = int(images.get_shape()[3])
    for i, next_conv_layer in enumerate(conv_layer):
        W_conv = weight_variables([5, 5, current_layer, next_conv_layer])
        b_conv = biases_variables([next_conv_layer])
        h_conv = conv2d(images_stream, W_conv, b_conv, [1, 2, 2, 1])
        images_stream = max_pool(h_conv, [1, 2, 2, 1], [1, 2, 2, 1])
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
            images_stream = batch_norm_wrapper(images_stream)
    return images_stream
```