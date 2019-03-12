import numpy as np
import os
import sys

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras import backend as K
from tensorflow.losses import huber_loss

sys.stderr = stderr

eps = np.float(K.epsilon())


def scaled_mean_squared_error(scale: float = 1.0, with_numpy: bool = False):
    """Scaled mean squared error

    Scaling is applied to the absolute error before squaring the data

    Keyword Arguments:
        scale {float} -- Scale factor (default: {1})

    Returns:
        {tensor} -- MSE of the scaled error
    """

    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square((y_pred - y_true) * scale), axis=-1)

    def mean_squared_error_numpy(y_true, y_pred):
        return np.mean(np.square((y_pred - y_true) * scale), axis=-1)

    if with_numpy:
        return mean_squared_error_numpy

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
        return huber_loss(y_true * scale, y_pred * scale, delta=delta)

    return huber


def binary_crossentropy_numpy(y_true, y_pred):
    output = np.clip(y_pred, eps, 1 - eps)

    return np.mean(
        -(y_true * np.log(output) + (1 - y_true) * np.log(1 - output)), axis=-1
    )


def get_loss(loss: str):
    loss_parts = loss.split("-")

    if loss.startswith("smse") and len(loss_parts) > 1:
        loss = scaled_mean_squared_error(float(loss_parts[1]))

    elif loss.startswith("smae") and len(loss_parts) > 1:
        loss = scaled_mean_absolute_error(float(loss_parts[1]))

    elif loss.startswith("shuber") and len(loss_parts) > 2:
        loss = scaled_huber(float(loss_parts[1]), float(loss_parts[2]))

    elif loss.startswith("slogcosh") and len(loss_parts) > 1:
        loss = scaled_logcosh(float(loss_parts[1]))

    elif loss.startswith("bce"):
        loss = "binary_crossentropy"

    return loss
