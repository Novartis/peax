"""
Copyright 2018 Novartis Institutes for BioMedical Research Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hnswlib
import importlib
import itertools
import numpy as np
import operator
import os
import sys
import warnings

from contextlib import contextmanager

from scipy.ndimage.interpolation import zoom
from scipy.stats import norm
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from typing import Callable, List

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import keras
from keras.layers import Input
from keras.models import Model

sys.stderr = stderr

flatten = itertools.chain.from_iterable


def compare_lists(
    a: List, b: List, conditionator: Callable = all, comparator: Callable = operator.eq
):
    return conditionator(map(comparator, a, itertools.islice(a, 1, None)))


def unpredictability(p: np.ndarray) -> float:
    """Unpredictability score

    Unpredictability is defined as the minimum deviation of the prediction probability
    from `0.5` to `0` or `1`. For example, for a prediction probability of 0.6 the
    unpredictability is 0.4. The highest unpredictability is 1 and the lowest is 0.
    """
    return np.mean(np.abs(p - np.round(p))) * 2


def prediction_proba_change(p0: np.ndarray, p1: np.ndarray) -> float:
    """Unpredictability score

    Total amount of change in the prediction probability
    """
    return np.mean(np.abs(p0 - p1))


def prediction_change(p0: np.ndarray, p1: np.ndarray, border: float = 0.5) -> float:
    """Prediction change score

    Prediction change is defined as the number of times the predicted class changes
    based on the border probability.
    """
    return np.mean(np.sign(p0 - border) != np.sign(p1 - border))


# def uncertainty(model, X_train: np.ndarray, X_test: np.ndarray) -> float:
#     """Unpredictability score
#
#     Unpredictability is defined as the minimum deviation of the prediction probability
#     from `0.5` to `0` or `1`. For example, for a prediction probability of 0.6 the
#     unpredictability is 0.4. The highest unpredictability is 1 and the lowest is 0.
#     """
#     return random_forest_error(model, X_train, X_test).mean()


def convergence(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, decimals: int = 2
) -> float:
    """Convergence score

    Given three measurements, the convergence score is the percentage of changes that
    increase or decrease in both steps. The highest convergence score is 1 and the
    lowest is 0.
    """
    x0r = np.round(x0, decimals=decimals)
    x1r = np.round(x1, decimals=decimals)
    x2r = np.round(x2, decimals=decimals)
    return np.mean(np.abs(np.sign(x1r - x0r) + np.sign(x2r - x1r)) == 2)


def divergence(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, decimals: int = 3
) -> float:
    """Divergence score

    Given three measurements, the divergence score is the percentage of changes that
    increase in one step and decrease in the other step or vice versa. The highest
    convergence score is 1 and the lowest is 0.
    """
    x0r = np.round(x0, decimals=decimals)
    x1r = np.round(x1, decimals=decimals)
    x2r = np.round(x2, decimals=decimals)
    d0 = np.sign(x1r - x0r)
    d1 = np.sign(x2r - x1r)
    return np.mean((d0 + d1 == 0) * (np.abs(d0) > 0))


def normalize(data, percentile: float = 99.9):
    cutoff = np.percentile(data, (0, percentile))
    data_norm = np.copy(data)
    data_norm[np.where(data_norm < cutoff[0])] = cutoff[0]
    data_norm[np.where(data_norm > cutoff[1])] = cutoff[1]

    return MinMaxScaler().fit_transform(data_norm)


def normalize_simple(data: np.ndarray):
    data -= np.min(data)
    return data / np.max(data)


def load_model(filepath: str, silent: bool = False, additional_args: list = None):
    try:
        if silent:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = keras.models.load_model(filepath)
        else:
            model = keras.models.load_model(filepath)
    except Exception:
        # We assume it's a custom model
        Model = getattr(
            importlib.import_module(os.path.dirname(filepath)),
            os.path.basename(filepath)
        )
        model = Model.load(*additional_args)

    return model


def get_encoder(autoencoder):
    # Find embedding layer
    embedding_layer_idx = None
    for i, layer in enumerate(autoencoder.layers):
        if layer.name == "embed":
            embedding_layer_idx = i

    # Create encoder
    inputs = autoencoder.input
    encoded = inputs
    for i in range(1, embedding_layer_idx + 1):
        encoded = autoencoder.layers[i](encoded)

    return Model(inputs, encoded)


def get_decoder(autoencoder):
    # Find embedding layer
    embedding_layer = None
    embedding_layer_idx = None
    for i, layer in enumerate(autoencoder.layers):
        if layer.name == "embed":
            embedding_layer = layer
            embedding_layer_idx = i

    embedding = embedding_layer.output_shape[1]

    encoded_input = Input(shape=(embedding,), name="input")
    decoded_input = encoded_input
    for i in range(embedding_layer_idx + 1, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)

    return Model(encoded_input, decoded_input)


def get_search_target_windows(
    db, search_id, window_size, abs_offset, no_stack: bool = False
):
    # Get search target window
    search = db.get_search(search_id)
    search_target_windows = get_target_window_idx(
        search["target_from"],
        search["target_to"],
        window_size,
        search["config"]["step_freq"],
        abs_offset,
    )

    # stwi == search target window indices
    stwi = np.arange(*search_target_windows[1])

    if no_stack:
        return stwi

    return np.hstack(
        (
            stwi.reshape(stwi.shape[0], 1),
            np.ones(stwi.shape[0]).reshape(stwi.shape[0], 1),
        )
    ).astype(int)


def get_search_target_classif(db, search_id, window_size, abs_offset):
    # Get search target window
    search = db.get_search(search_id)
    search_target_windows = get_target_window_idx(
        search["target_from"],
        search["target_to"],
        window_size,
        search["config"]["step_freq"],
        abs_offset,
    )

    # stwi == search target window indices
    stwi = np.arange(*search_target_windows[1])
    return np.hstack(
        (
            stwi.reshape(stwi.shape[0], 1),
            np.ones(stwi.shape[0]).reshape(stwi.shape[0], 1),
        )
    ).astype(int)


def get_num_windows(chrom_size, window_size, step_size):
    return np.ceil((chrom_size - window_size) / step_size).astype(int) + 1


def scaleup_vector(v, out_len, aggregator: Callable = np.mean):
    in_len = v.shape[0]
    lcm = np.lcm(in_len, out_len)
    blowup = np.repeat(v, lcm / in_len)
    return aggregator(blowup.reshape(-1, (lcm / out_len).astype(int)), axis=1)


def zoom_array(
    in_array,
    final_shape,
    same_sum=False,
    aggregator=np.mean,
    zoomor=zoom,
    **zoomor_kwargs
):
    """Rescale vectors savely.

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.
    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?
    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.
    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    Parameters
    ----------
    in_array: n-dimensional numpy array (1D also works)
    final_shape: resulting shape of an array
    same_sum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    aggregator: by default, np.mean. You can plug your own.
    zoomor: by default, scipy.ndimage.zoom. You can plug your own.
    zoomor_kwargs:  a dict of options to pass to zoomor.
    """
    in_array = np.asarray(in_array, dtype=np.double)
    in_shape = in_array.shape

    assert len(in_shape) == len(final_shape), "Number of dimensions need to equal"

    mults = []  # multipliers for the final coarsegraining
    for i in range(len(in_shape)):
        if final_shape[i] < in_shape[i]:
            mults.append(int(np.ceil(in_shape[i] / final_shape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    temp_shape = tuple([i * j for i, j in zip(final_shape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoom_multipliers = np.array(temp_shape) / np.array(in_shape) + 0.0000001
    assert zoom_multipliers.min() >= 1

    # applying zoom
    rescaled = zoomor(in_array, zoom_multipliers, **zoomor_kwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1 :]
            rescaled.shape = newshape
            rescaled = aggregator(rescaled, axis=ind + 1)

    assert rescaled.shape == final_shape

    if same_sum:
        extra_size = np.prod(final_shape) / np.prod(in_shape)
        rescaled /= extra_size

    return rescaled


def merge_interleaved(v, step_freq, aggregator=np.nanmean):
    v_len = v.shape[0]
    out_len = v_len + (step_freq - 1)

    blowup = np.zeros((out_len, step_freq))
    blowup[:] = np.nan

    for i in np.arange(step_freq):
        blowup[:, i][i : min(i + v_len, out_len)] = v[: min(v_len, out_len - i)]

    return aggregator(blowup, axis=1)


def get_norm_sym_norm_kernel(size):
    half_a = np.ceil(size / 2).astype(int)
    half_b = np.floor(size / 2).astype(int)

    # Normal distribution from the 1st to the 99th percentile
    k = norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), size))

    # Normalize to 1
    k /= np.max(k)

    # Make symmetric to be usable for convex combination (e.g., in weighted
    # averaging)
    kn = k
    kn[:half_a] = k[:half_a] / (k[:half_a] + k[:half_a][::-1])
    kn[half_b:] = kn[:half_a][::-1]

    return kn


def merge_interleaved_mat(m: np.ndarray, step_freq: int, kernel: np.ndarray = None):
    if kernel is None:
        # Take the mean of the interleave vectors by default
        kernel = np.ones(m.shape[1])

    # length of one consecutive encoding
    M = np.int(m.shape[0] / step_freq) * m.shape[1]
    # Step size of windows
    # I.e., including binning, so 12Kb at 100 bins = 120 bin windows
    SZ = np.int(m.shape[1] / step_freq)
    # Out length
    # N = M + ((step_freq - 1) * SZ)
    # Out matrix
    o = np.zeros((M, step_freq))
    o[:] = np.nan
    # Kernel matrix
    k = np.zeros((M, step_freq))
    k[:] = np.nan
    long_k = np.tile(kernel, M)

    for i in np.arange(step_freq):
        # Linear, consecutive encoding
        LCE = m[i::step_freq].flatten()

        j = i * SZ

        o[:, i][j:M] = LCE[: M - j]
        k[:, i][j:M] = long_k[: M - j]

    # Normalize kernels
    k /= np.nansum(k, axis=1).reshape(k.shape[0], -1)

    return np.nansum(o * k, axis=1)


def hashify(l: list, key: str) -> dict:
    h = {}
    for item in l:
        key_value = item.get(key, "unknown")
        h[key_value] = item
    return h


def is_int(s: str, is_pos: bool) -> bool:
    if s is None:
        return False
    try:
        i = int(s)
        if is_pos:
            return i >= 0
        return True
    except ValueError:
        return False


def kNN(data: np.ndarray, id: int, n: int) -> np.ndarray:
    dist = np.sqrt(np.sum((data - data[id]) ** 2, axis=1))
    return np.argsort(dist)[1 : n + 1]


def enforce_window_size(start, end, window_size):
    if end - start == window_size:
        return np.array([start, end])

    size = end - start
    center = start + (size // 2)
    return np.array([center - window_size // 2, center + window_size // 2])


def serialize_classif(classif):
    sorting = np.argsort(classif[:, 0])
    merged = classif[:, 0] * classif[:, 1]
    return merged[sorting].tobytes()


def unserialize_classif(serialized_classif):
    return np.frombuffer(serialized_classif, dtype=np.int)


def impact(data, impact=1.0):
    impact = min(1, max(0, impact))
    return impact * data + (1 - impact)


def get_target_window_idx(
    target_from: int,
    target_to: int,
    window_size: int,
    step_freq: int,
    abs_offset: int,
    max_offset: float = 0.66,
) -> list:
    step_size = window_size / step_freq
    target_locus = enforce_window_size(target_from, target_to, window_size)
    target_locus[0] -= abs_offset
    target_locus[1] -= abs_offset
    window_from_idx = int(target_locus[0] // step_size)
    window_from_pos = int(window_from_idx * step_size)
    window_to_idx = window_from_idx + step_freq

    # Remove windows that overlap too much with the target search
    offset = (target_locus[0] - window_from_pos) / window_size
    k = step_freq * (offset - max_offset)
    m = np.ceil(k).astype(int)
    n = step_freq * offset

    return (
        # Including any kind of overlaping window
        (window_from_idx + np.floor(k), window_to_idx + np.ceil(n)),
        # Only include windows that overlap at least 33% with the target
        (window_from_idx + m, window_to_idx + m),
    )


def knn_density(
    data: np.ndarray,
    k: int = 5,
    dist_metric: str = "euclidean",
    summary: Callable[[np.ndarray], np.float64] = np.mean,
):
    n, dim = data.shape

    if (n > 100000):
        # Declaring index
        p = hnswlib.Index(space='l2', dim=dim)

        # Also see https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        ef = np.int(np.ceil(20 * np.log2(n)))

        # Initing index - the maximum number of elements should be known beforehand
        p.init_index(max_elements=n, ef_construction=ef, M=16)

        # Element insertion (can be called several times):
        p.add_items(data, np.arange(n))

        # Controlling the recall by setting ef
        p.set_ef(ef)

        _, dist = p.knn_query(data, k = k)

        # Delete the index
        del p
    else:
        leaf_size = np.int(np.round(10 * np.log(n)))
        bt = BallTree(data, leaf_size=leaf_size)
        dist, _ = bt.query(data, k, dualtree=True, sort_results=False)

    try:
        return summary(dist, axis=1)
    except Exception:
        out = np.zeros(dist.shape[0])
        out[:] = np.nan
        return out


@contextmanager
def suppress_with_default(*exceptions, **kwargs):
    """Like contextlib.suppress but with a default value on exception

    Decorators:
        contextmanager

    Arguments:
        *exceptions {list} -- List of exceptions to suppress. By default all exceptions are suppressed.
        **kwargs {dict} -- Dictionary of key word arguments

    Yields:
        any -- Default value from ``kwargs``
    """
    try:
        yield kwargs.get("default", None)
    except exceptions or Exception:
        pass


def get_c(target_c: list, bg_c: list, opacity: float):
    target = np.array(target_c) / 255
    bg = np.array(bg_c) / 255
    return ((target * (1 / opacity) - bg * ((1 - opacity) / opacity)) * 255).astype(int)
