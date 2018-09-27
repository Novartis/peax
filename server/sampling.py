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


def random_sampling(data: np.ndarray, num_samples: int = 20):
    try:
        return data[
            np.random.choice(data.shape[0], size=num_samples, replace=False)
        ]
    except ValueError as e:
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
    except ValueError as e:
        dist = cdist(
            data[selected], target.reshape((1, target.size))
        ).flatten()

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
    seeds[half:] = random_sampling(
        np.where(selected)[0], num_samples=other_half
    )
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
        except ValueError as e:
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
    target: np.ndarray,
    n: int = 5,
    d: float = 0.5,
    dist_metric: str = "euclidean",
):
    # Select regions
    sdata = data[selected]
    indices = np.where(selected)[0]
    try:
        dist = cdist(
            sdata, target.reshape((1, target.size)), dist_metric
        ).flatten()
    except ValueError as e:
        dist = cdist(sdata, target.reshape((1, target.size))).flatten()

    # Calculate the point density of using the pairwise (pw) distances of the
    # k nearest neighbors
    pw_knn_density = utils.knn_density(sdata)

    # Get the distance at half of the popuation
    centered_dist = np.abs(dist - np.median(dist))
    centered_knn_density = (centered_dist / np.max(centered_dist)) / (
        pw_knn_density / np.max(pw_knn_density)
    )

    centered_knn_density_sorted_idx = np.argsort(centered_knn_density)

    return indices[centered_knn_density_sorted_idx][:n]


def sample_by_uncertainty_density(
    data: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    p_y: np.ndarray,
    n: int = 5,
    k: int = 5,
    d_weight: float = 0.5,
    dist_metric: str = "euclidean",
):
    # Select regions
    sdata = data[selected]
    indices = np.where(selected)[0]

    # Convert the class probabilities into probabilities of uncertainty
    # p = 0:   1 - abs(0   - 0.5) * 2 == 0
    # p = 0.5: 1 - abs(0.5 - 0.5) * 2 == 1
    # p = 1:   1 - abs(1   - 0.5) * 2 == 0
    p_uncertain = 1 - np.abs(p_y[selected] - 0.5) * 2

    # Calculate the point density of using the pairwise (pw) distances of the
    # k nearest neighborse
    pw_knn_density = utils.knn_density(sdata, k=k)

    # Normalize
    pw_knn_density /= np.max(pw_knn_density)
    # Weight density: a value of 0.5 downweights the importants by scaling
    # the relative density closer to 1
    pw_knn_density = pw_knn_density * d_weight + (1 - d_weight)

    # Combine uncertainty and density
    uncertain_knn_density = p_uncertain * pw_knn_density

    # Sort descending
    uncertain_knn_density_sorted_idx = np.argsort(uncertain_knn_density)[::-1]

    return indices[uncertain_knn_density_sorted_idx][:n]
