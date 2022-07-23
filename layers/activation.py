# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

import tensorflow as tf

from tensorflow.python.keras.layers import Layer, Activation

try:
    unicode
except NameError:
    unicode = str


def activation_layer(activation):
    if isinstance(activation, (str, unicode)):
        act_layer = Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer
