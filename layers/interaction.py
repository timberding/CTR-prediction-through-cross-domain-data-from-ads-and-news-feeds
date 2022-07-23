# -*- coding:utf-8 -*-
"""

Authors:
    Weichen Shen,weichenswc@163.com,
    Harshit Pande

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

try:
    from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
        glorot_normal_initializer as glorot_normal, \
        glorot_uniform_initializer as glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, Constant, TruncatedNormal, glorot_normal, glorot_uniform

from tensorflow.python.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten
from tensorflow.python.keras.regularizers import l2


class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Arguments
        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.

        - **seed**: A Python integer to use as random seed.

      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.parameterization == 'matrix':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, dim),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'layer_num': self.layer_num, 'parameterization': self.parameterization,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNet, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape

