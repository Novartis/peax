import os
import sys

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras import backend as K
from tensorflow.losses import huber_loss

sys.stderr = stderr


def scaled_mean_squared_error(scale: float = 1.0):
    """Scaled mean squared error

    Scaling is applied to the absolute error before squaring the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- MSE of the scaled error
    """

    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true) * scale), axis=-1)

    return mean_squared_error


def scaled_mean_absolute_error(scale: float = 1.0):
    """Scaled mean absolute error

    Scaling is applied to the absolute error before taking the absolute the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- MAE of the scaled error
    """

    def mean_absolute_error(y_true, y_pred):
        return K.mean(K.abs((y_pred - y_true) * scale), axis=-1)

    return mean_absolute_error


def scaled_logcosh(scale: float = 1.0):
    """Scale logcosh loss

    Scaling is applied to the absolute error before logcoshing the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- Logcosh of the scaled error
    """

    def _logcosh(x):
        return x + K.softplus(-2.0 * x) - K.log(2.0)

    def logcosh(y_true, y_pred):
        return K.mean(_logcosh((y_pred - y_true) * scale), axis=-1)

    return logcosh


def scaled_huber(scale: float = 1.0, delta: float = 1.0):
    """Scaled Huber loss

    Scaling is applied to the absolute error before hubering the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})
        delta {float} -- Huber's delta parameter (default: {1})

    Returns:
        {tensor} -- Huber loss of the scaled error
    """

    def huber(y_true, y_pred):
        return K.mean(huber_loss(y_true * scale, y_pred * scale, delta=delta), axis=-1)

    return huber
