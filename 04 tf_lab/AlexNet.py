
import tensorflow as tf
import numpy as np

"""
    AlexNet
    - Implementation of AlexNet
    - Reference: Alex Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks"
"""
class AlexNet(object):

    def __init__(self, x, dropout_prob, num_classes):

        self.X = x
        self.NUM_CLASSES = num_classes
        self.DROPOUT_PROB = dropout_prob

        self.build_net()

    def build_net(self):
        # 1st Layer
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-04, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-04, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fully_connected(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.DROPOUT_PROB)

        # 7th Layer
        fc7 = fully_connected(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.DROPOUT_PROB)

        # 8th Layer
        self.fc8 = fully_connected(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='VALID', groups=1):
    input_channels = int(x.get_shape()[-1])

    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        conv = tf.concat(axis=3, values=output_groups)

    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    relu = tf.nn.relu(bias, name=scope.name)

    return relu

"""
    Fully connected layer
"""
def fully_connected(x, n_in, n_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable(
            'weights',
            shape=[n_in, n_out],
            trainable=True
        )
        biases = tf.get_variable('biases', [n_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


"""
    Max pooling
"""
def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='VALID'):
    return tf.nn.max_pool(
        x,
        ksize=[1, filter_height, filter_width, 1],
        strides=[1, stride_y, stride_x, 1],
        padding=padding,
        name=name
    )

"""
    Local response normalization
"""
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(
        x,
        depth_radius=radius,
        alpha=alpha,
        beta=beta,
        bias=bias,
        name=name
    )

def dropout(x, dropout_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, dropout_prob)
