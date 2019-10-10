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

import numpy as np
from scipy.spatial.distance import cdist
from server import utils

from numba import njit
from numba import prange


def random_sampling(data: np.ndarray, num_samples: int = 20):
    try:
        return data[np.random.choice(data.shape[0], size=num_samples, replace=False)]
    except ValueError:
        print(
            "WARNING: too few data points ({}) for sampling {} items".format(
                data.shape[0], num_samples
            )
        )
        return data


def dist_sampling(
    data: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    num_samples: int = 20,
    dist_metric: str = "euclidean",
):
    indices = np.where(selected)[0]
    try:
        dist = cdist(
            data[selected], target.reshape((1, target.size)), dist_metric
        ).flatten()
    except ValueError:
        dist = cdist(data[selected], target.reshape((1, target.size))).flatten()

    dist_sorted_idx = np.argsort(dist)
    indices_sorted = indices[dist_sorted_idx]

    N = dist.size

    # Number of ranges such that at least 10 windows are to be chosen from
    n = np.floor(np.log2(N / 10)).astype(int)
    samples_per_round = np.floor(num_samples / n).astype(int)
    samples_last_round = num_samples - samples_per_round * (n - 1)
    samples_per_round = [samples_per_round] * (n - 1) + [samples_last_round]

    # Lets sample from ranges of halfing size. E.g.: the * mark the range
    # |*|**|****|********|
    samples = np.zeros(num_samples)
    start = np.round(N / 2).astype(int)
    end = N
    k = 0
    for i in np.arange(n):
        # Randomly sample points from the range
        samples[k : k + samples_per_round[i]] = random_sampling(
            indices_sorted[start:end], num_samples=samples_per_round[i]
        )
        end = start
        start = np.round(start / 2).astype(int)
        k += samples_per_round[i]

    return samples


def get_seeds(
    data: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    num_seeds: int = 40,
    dist_metric: str = "euclidean",
):
    half = int(num_seeds // 2)
    other_half = num_seeds - half
    seeds = np.zeros(num_seeds).astype(int)

    # First half is sampled in increasing distance to the target
    seeds[:half] = dist_sampling(
        data, selected, target, dist_metric=dist_metric, num_samples=half
    )
    # data_no_dist = np.delete(data, seeds[0:half], axis=0)
    selected[seeds[0:half]] = False

    # Half of the seeds are sampled randomly
    seeds[half:] = random_sampling(np.where(selected)[0], num_samples=other_half)
    # data_no_dist_rnd = np.detele(data, samples[0:half * 2], axis=0)

    # The other half are sampled spatially randomly
    # samples[third * 2:] = spatial_sampling(
    #   data_no_dist_rnd, num_samples=third
    # )

    return seeds


def seeds_by_dim(
    data: np.ndarray, src: np.ndarray, metric: str = "euclidean", sdim: int = 4
):
    N = data.shape[0]
    ndim = src.ndim

    samples = np.empty((sdim * ndim,)).astype(int)

    for dim in range(ndim):
        if np.max(data[:, dim]) == 0:
            print("Skipping meaningless dimension")
            continue

        # Remove already sampled points (nr = non-redundant)
        nr_data = np.delete(data, samples[: dim * sdim], 0)

        dim_data = nr_data[:, dim].reshape((nr_data.shape[0], 1))
        dim_src = src[dim].reshape((1, 1))
        reduced_data = np.delete(nr_data, dim, 1)
        reduced_src = np.delete(src, dim, 0).reshape((1, src.size - 1))

        try:
            dim_dist = cdist(dim_data, dim_src, metric).flatten()
            reduced_dist = cdist(reduced_data, reduced_src, metric).flatten()
        except ValueError:
            dim_dist = cdist(dim_data, dim_src).flatten()
            reduced_dist = cdist(reduced_data, reduced_src).flatten()

        max_dim_dist = np.max(dim_dist)

        n = 25
        dim_var = 0
        while dim_var < 0.1 and n < N:
            dim_sample = dim_dist[np.argsort(reduced_dist)[:n]]
            dim_var = np.max(dim_sample) / max_dim_dist
            n *= 2

        # Random sample
        dim_sample2 = np.argsort(dim_sample)
        np.random.randint(sdim, size=dim_sample2.size)

        print(
            dim_sample2.shape[0],
            nr_data.shape[0],
            max_dim_dist,
            dim * sdim,
            (dim + 1) * sdim,
            dim_sample2.size,
        )

        samples[dim * sdim : (dim + 1) * sdim] = np.random.randint(
            sdim, size=dim_sample2.size
        )


def sample_by_dist_density(
    data: np.ndarray,
    selected: np.ndarray,
    dist_to_target: np.ndarray,
    knn_density: np.ndarray,
    levels: int = 5,
    level_sample_size: int = 5,
    initial_level_size: int = 10,
    dist_metric: str = "euclidean",
):
    """Sample by distance and density

    This sampling strategy is based on the distance of the encoded windows to the
    encoded target window and the windows' knn-density. For `levels` number of
    increasing size it samples iteratively by density and maximum distance to already
    sampled windows. Essentially, this search strategy samples in increases radii. You
    can think of it like this:

    |  ....       ..   .  . |.. . . |. . .[X]  . |. ...|   ...  . .  ...|
    2    ↑                  1 ↑     0  ↑       ↑ 0   ↑ 1     ↑          3

    Where `[X]` is the search target, `.` are windows, and `|` indicate the level
    boundaries, and `↑` indicates the sampled windows. The boundaries are defined by
    the number of windows for a certain level. In this examples, the
    `initial_level_size` is {4}, so the first level consists of the 4 nearest neighbors
    to the search target. The second level consists of the next 8 nearest neighbors,
    the third level consists of the next 16 nearest neighbors, etc. Within these levels
    the algorithm iteratively samples the densest windows that are farthest away from
    the already sampled windows. E.g., in level zero the left sample is selected because
    it's close to other windows but the second sample is selected because it's far away
    from the first sample while the other available windows are too close to the already
    sampled window.

    Arguments:
        data {np.ndarray} -- The complete data
        selected {np.ndarray} -- A subset of the data to be sampled on
        dist_to_target {np.ndarray} -- Pre-computed distances between each data item
            (i.e., row in `data`) and the search target in the latent space
        knn_density {np.ndarray} -- Pre-computed knn-density of every data item
            (i.e., row in `data`) in the latent space

    Keyword Arguments:
        levels {int} -- The number of levels used for sampling. The final number of
            sampled windows will be `levels` times `levels_sample_size` (default: {5})
        level_sample_size {int} -- The number of windows to be sampled per level.
            (default: {5})
        initial_level_size {int} -- The number of windows considered to be the first
            level for which `level_sample_size` windows are sampled. In every subsequent
            sampling the size is doubled. (default: {10})
        dist_metric {str} -- The distance metric used to determine the distance between
            already sampled windows and the remaining windows in the level.
            (default: {"euclidean"})

    Returns:
        {np.ndarray} -- Sampled windows
    """
    sdata = data[selected]
    selected_idx = np.where(selected)[0]

    dist_selected = dist_to_target[selected]
    knn_density_selected = knn_density[selected]

    rel_wins_idx_sorted_by_dist = np.argsort(dist_selected)

    all_samples = np.zeros(levels * level_sample_size).astype(np.int)

    from_size = 0
    for l in range(levels):
        to_size = from_size + initial_level_size * (2 ** l)

        # Select all windows in an increasing distance from the search target
        rel_wins_idx = rel_wins_idx_sorted_by_dist[from_size:to_size]
        selection_mask = np.zeros(rel_wins_idx.size).astype(np.bool)

        # Get the sorted list of windows by density
        # Note that density is determined as the average distance of the 5 nearest
        # neighbors of the window. Hence, the lowest value indicates the highest
        # density. The absolute minimum is 0, when all 5 nearest neighbors are the same
        wins_idx_by_knn_density = np.argsort(knn_density_selected[rel_wins_idx])

        samples = np.zeros(level_sample_size).astype(np.uint32) - 1

        # Sample first window by density only
        samples[0] = wins_idx_by_knn_density[0]
        selection_mask[samples[0]] = True

        for s in range(1, level_sample_size):
            # Get all the windows in the current level that have not been sampled yet
            remaining_wins_idx = rel_wins_idx[~selection_mask]
            remaining_wins = sdata[remaining_wins_idx]
            sampled_wins = sdata[samples[:s]]

            if s == 1:
                sampled_wins = sampled_wins.reshape((1, -1))

            # Get the normalized summed distance of the remaining windows to the
            # already sampled windows
            dist = np.sum(
                utils.normalize(cdist(remaining_wins, sampled_wins, dist_metric)),
                axis=1,
            )

            # Select the window that is farthest away from the already sampled windows
            # and is in the most dense areas
            samples[s] = np.where(~selection_mask)[0][
                np.argsort(
                    utils.normalize_simple(np.max(dist) - dist)
                    + utils.normalize_simple(knn_density_selected[remaining_wins_idx])
                )[0]
            ]
            selection_mask[samples[s]] = True

        all_samples[l * level_sample_size : (l + 1) * level_sample_size] = rel_wins_idx[
            samples
        ]

        from_size = to_size

    return selected_idx[all_samples]


def maximize_pairwise_distance(
    data: np.ndarray,
    ranked_candidates: np.ndarray,
    rank_values: np.ndarray,
    n: int,
    dist_metric: str = "euclidean",
    dist_aggregator: callable = np.mean,
):
    samples = np.zeros(n).astype(np.uint32) - 1
    mask = np.zeros(ranked_candidates.size).astype(np.bool)

    samples[0] = ranked_candidates[0]
    mask[0] = True

    d = cdist(data[ranked_candidates], data[ranked_candidates], dist_metric)
    d = d.max() - d
    d -= d.min()
    d /= d.max()
    d *= utils.normalize_simple(rank_values)

    for i in np.arange(1, n):
        # Get all the windows in the current level that have not been sampled yet
        remaining_wins_idx = ranked_candidates[~mask]
        remaining_wins = data[remaining_wins_idx]
        sampled_wins = data[samples[:i]]

        if i == 1:
            sampled_wins = sampled_wins.reshape((1, -1))

        # Get the normalized summed distance of the remaining windows to the
        # already sampled windows
        dist = dist_aggregator(
            utils.normalize(cdist(remaining_wins, sampled_wins, dist_metric)), axis=1
        )

        dist_rank_sorted = np.argsort(
            utils.normalize_simple(np.max(dist) - dist)
            + utils.normalize_simple(rank_values[~mask])
        )

        # Select the window that is farthest away from the already sampled windows
        # and is in the most dense areas
        rel_idx = np.where(~mask)[0][dist_rank_sorted[0]]
        mask[rel_idx] = True
        samples[i] = ranked_candidates[rel_idx]

    return samples


@njit(
    "int64(float64[:,:], float64[:], float64[:], boolean[:])", nogil=True, parallel=True
)
def compute_gains(X, gains, current_values, mask):
    for idx in prange(X.shape[0]):
        if mask[idx] == 1:
            continue

        gains[idx] = np.maximum(X[idx], current_values).sum()

    return np.argmax(gains)


def weighted_facility_locator(
    data: np.ndarray,
    ranked_candidates: np.ndarray,
    rank_values: np.ndarray,
    n: int,
    dist_metric: str = "euclidean",
):
    num_candidates = ranked_candidates.shape[0]
    samples = np.zeros(n).astype(np.uint32) - 1
    mask = np.zeros(num_candidates).astype(np.bool)

    samples[0] = ranked_candidates[0]
    mask[0] = True

    d = cdist(data[ranked_candidates], data[ranked_candidates], dist_metric)
    d = d.max() - d
    d /= d.max()
    d *= utils.normalize_simple(rank_values).reshape((-1, 1))

    current_values = np.zeros(num_candidates, dtype="float64")

    for i in np.arange(1, n):
        gains = np.zeros(num_candidates, dtype="float64")

        compute_gains(d, gains, current_values, mask)

        gains -= current_values.sum()

        best_candidate_idx = gains.argmax()

        current_values = np.maximum(d[best_candidate_idx], current_values).astype(
            "float64"
        )

        samples[i] = ranked_candidates[best_candidate_idx]
        mask[best_candidate_idx] = True

    return samples


def sample_by_uncertainty_dist_density(
    data: np.ndarray,
    selected: np.ndarray,
    dist_to_target: np.ndarray,
    knn_dist: np.ndarray,
    p_y: np.ndarray,
    n: int = 10,
    k: int = 5,
    knn_dist_weight: float = 0.5,
    dist_to_target_weight: float = 0.5,
    dist_metric: str = "euclidean",
):
    # Select regions
    indices = np.where(selected)[0]

    # Convert the class probabilities into probabilities of uncertainty
    # p = 0:   1 - abs(0   - 0.5) * 2 == 0
    # p = 0.5: 1 - abs(0.5 - 0.5) * 2 == 1
    # p = 1:   1 - abs(1   - 0.5) * 2 == 0
    p_uncertain = 1 - np.abs(p_y[selected] - 0.5) * 2

    # Normalize knn-density
    knn_dist_norm_selected = utils.normalize_simple(knn_dist[selected])
    # Weight density: a value of 0.5 down-weights the importance by scaling
    # the relative density closer to 1
    knn_dist_norm_selected_weighted = utils.impact(
        knn_dist_norm_selected, knn_dist_weight
    )
    knn_dist_norm_selected_weighted = (
        knn_dist_norm_selected_weighted.max() - knn_dist_norm_selected_weighted
    )

    # Normalize distance to target
    dist_to_target_norm_selected = utils.normalize_simple(dist_to_target[selected])
    dist_to_target_norm_selected_weighted = utils.impact(
        dist_to_target_norm_selected, dist_to_target_weight
    )
    dist_to_target_norm_selected_weighted = (
        dist_to_target_norm_selected_weighted.max()
        - dist_to_target_norm_selected_weighted
    )

    # Combine uncertainty, knn distance, and distance to target
    # The value is in [0, 2]
    uncertain_knn_dist = p_uncertain * (
        p_uncertain
        + knn_dist_norm_selected_weighted
        + dist_to_target_norm_selected_weighted
    )
    uncertain_knn_dist /= uncertain_knn_dist.max()

    # Sort descending (highest is best!)
    uncertain_knn_dist_sorted_idx = np.argsort(uncertain_knn_dist)[::-1]

    # Subsample by maximizing the pairwise distance
    subsample = maximize_pairwise_distance(
        data,
        indices[uncertain_knn_dist_sorted_idx][: n * 10],
        uncertain_knn_dist[uncertain_knn_dist_sorted_idx][: n * 10],
        n,
    )

    return subsample
