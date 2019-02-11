import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from keras import backend as K


def dtw_metric(dist: callable = euclidean, radius: int = 1):
    def dtw(y_true, y_pred):
        d = np.zeros(y_true.shape[0])

        for i in np.arange(y_true.shape[0]):
            d[i] = fastdtw(y_true[i, :], y_pred[i, :], dist=dist, radius=radius)[0]

        return d

    return dtw


def r2_min(y_true, y_pred):
    """R Squared (minimum)

    R Sqaured (minimum) lies between 0 and infinity where 0 is best!
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / (SS_tot + K.epsilon())


def r2(y_true, y_pred):
    """R Squared

    R Sqaured lies between negative infinity and 1. 1 is best!
    """
    return 1 - r2_min(y_true, y_pred)
