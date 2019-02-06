import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw_metric(dist: callable = euclidean, radius: int = 1):
    def dtw(y_true, y_pred):
        d = np.zeros(y_true.shape[0])

        for i in np.arange(y_true.shape[0]):
            d[i] = fastdtw(y_true[i, :], y_pred[i, :], dist=dist, radius=radius)[0]

        return d

    return dtw
