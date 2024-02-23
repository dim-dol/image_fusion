import tensorflow as tf
import numpy as np

WEIGHT_INIT_STDDEV = 0.1

class adaptive_densenet():
    def __init__(self):
        self.weight_vars = []

        with tf.compat.v1.variable_scope('adaptive_densenet'):
            self.weight_vars.append(self._create_variables(4, 16, 3, scope='conv_0'))
            self.weight_vars.append(self._create_variables(16, 16, 3, scope='dense_1'))
            self.weight_vars.append(self._create_variables(32, 16, 3, scope='dense_2'))
            self.weight_vars.append(self._create_variables(48, 16, 3, scope='dense_3'))
            self.weight_vars.append(self._create_variables(64, 16, 3, scope='dense_4'))
            self.weight_vars.append(self._create_variables(80, 16, 3, scope='dense_5'))
            self.weight_vars.append(self._create_variables(96, 32, 3, scope='conv_1'))
            self.weight_vars.append(self._create_variables(32, 3, 3, scope='conv'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.compat.v1.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')

        return (kernel, bias)

    def dense_net(self, image):
        final_layer_idx = len(self.weight_vars) -1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            # conv
            if i == final_layer_idx:
                out = convblock(out, kernel, bias, activate=False)

            # conv_0, conv_1
            elif i == 0 or i == final_layer_idx - 1:
                out = convblock(out, kernel, bias)

            # others
            else:
                out = denseblock(out, kernel, bias)

        return out


def convblock(c, kernel, bias, activate=True):

    if activate:
        c = tf.pad(c, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")

    c = tf.nn.conv2d(c, kernel, strides=[1, 1, 1, 1], padding='VALID')
    c = tf.nn.bias_add(c, bias)

    if activate:
        c = tf.nn.leaky_relu(c)
    else:
        c = tf.nn.tanh(c)

    return c


def denseblock(input, kernel, bias):
    c = convblock(input, kernel, bias)
    c = tf.concat([c, input], axis=-1)
    return c
