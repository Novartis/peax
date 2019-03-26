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

import base64
import bbi
import cooler
import numpy as np
import os
import pandas as pd

from server import utils


TILE_SIZE = 1024

TILESET_INFO = {"filetype": "bigwig", "datatype": "vector"}

FILE_EXT = {"bigwig", "bw"}


def is_bigwig(filepath=None, filetype=None):
    if filepath is None:
        return False

    if filetype == "bigwig":
        return True

    filename, file_ext = os.path.splitext(filepath)

    if file_ext[1:].lower() in FILE_EXT:
        return True

    return False


def get_quadtree_depth(chromsizes):
    tile_size_bp = TILE_SIZE
    min_tile_cover = np.ceil(sum(chromsizes) / tile_size_bp)
    return int(np.ceil(np.log2(min_tile_cover)))


def get_zoom_resolutions(chromsizes):
    return [2 ** x for x in range(get_quadtree_depth(chromsizes) + 1)][::-1]


def get_chromsizes(bwpath):
    """TODO: replace this with negspy.

    Also, return NaNs from any missing chromosomes in bbi.fetch
    """
    chromsizes = bbi.chromsizes(bwpath)
    chromosomes = cooler.util.natsorted(chromsizes.keys())
    return pd.Series(chromsizes)[chromosomes]


def chr2abs(chromsizes, chr: str, start: int, end: int):
    """Convert chromosomal coordinates to absolute coordinates.

    Arguments:
        chromsizes -- [description]
        chr -- [description]
        start -- [description]
        end -- [description]

    Yields:
        [type] -- [description]
    """
    offset = (np.cumsum(chromsizes) - chromsizes)[chr]
    return (offset + start, offset + end)


def abs2chr(chromsizes, start_pos: int, end_pos: int, is_idx2chr: bool = False):
    """Convert absolute coordinates to chromosomal coordinates.

    Arguments:
        chromsizes {[type]} -- [description]
        start_pos {[type]} -- [description]
        end_pos {[type]} -- [description]

    Yields:
        [type] -- [description]
    """
    abs_chrom_offsets = np.r_[0, np.cumsum(chromsizes.values)]
    cid_lo, cid_hi = (
        np.searchsorted(abs_chrom_offsets, [start_pos, end_pos], side="right") - 1
    )
    rel_pos_lo = start_pos - abs_chrom_offsets[cid_lo]
    rel_pos_hi = end_pos - abs_chrom_offsets[cid_hi]
    start = rel_pos_lo

    def idx2chr(cid):
        return chromsizes.index[cid] if is_idx2chr else cid

    for cid in range(cid_lo, cid_hi):
        yield idx2chr(cid), start, chromsizes[cid]
        start = 0
    yield idx2chr(cid_hi), start, rel_pos_hi


def get_tile(bwpath, zoom_level, start_pos, end_pos, chromsizes=None):
    if chromsizes is None:
        chromsizes = get_chromsizes(bwpath)

    resolutions = get_zoom_resolutions(chromsizes)
    binsize = resolutions[zoom_level]

    arrays = []
    for cid, start, end in abs2chr(chromsizes, start_pos, end_pos):
        n_bins = int(np.ceil((end - start) / binsize))
        try:
            chrom = chromsizes.index[cid]
            clen = chromsizes.values[cid]

            x = bbi.fetch(bwpath, chrom, start, end, bins=n_bins, missing=np.nan)

            # drop the very last bin if it is smaller than the binsize
            if end == clen and clen % binsize != 0:
                x = x[:-1]
        except IndexError:
            # beyond the range of the available chromosomes
            # probably means we've requested a range of absolute
            # coordinates that stretch beyond the end of the genome
            x = np.zeros(n_bins)

        arrays.append(x)

    return np.concatenate(arrays)


def tiles(bwpath, tile_ids, chromsizes=None):
    """Generate tiles from a bigwig file.

    Parameters
    ----------
    tileset: tilesets.models.Tileset object
        The tileset that the tile ids should be retrieved from
    tile_ids: [str,...]
        A list of tile_ids (e.g. xyx.0.0) identifying the tiles
        to be retrieved

    Returns
    -------
    tile_list: [(tile_id, tile_data),...]
        A list of tile_id, tile_data tuples
    """
    generated_tiles = []
    for tile_id in tile_ids:
        tile_id_parts = tile_id.split(".")
        tile_position = list(map(int, tile_id_parts[1:3]))
        zoom_level = tile_position[0]
        tile_pos = tile_position[1]

        if chromsizes is None:
            chromsizes = get_chromsizes(bwpath)

        max_depth = get_quadtree_depth(chromsizes)
        tile_size = TILE_SIZE * 2 ** (max_depth - zoom_level)
        start_pos = tile_pos * tile_size
        end_pos = start_pos + tile_size
        dense = get_tile(bwpath, zoom_level, start_pos, end_pos, chromsizes=chromsizes)

        if len(dense):
            max_dense = max(dense)
            min_dense = min(dense)
        else:
            max_dense = 0
            min_dense = 0

        min_f16 = np.finfo("float16").min
        max_f16 = np.finfo("float16").max

        has_nan = len([d for d in dense if np.isnan(d)]) > 0

        if (
            not has_nan
            and max_dense > min_f16
            and max_dense < max_f16
            and min_dense > min_f16
            and min_dense < max_f16
        ):
            tile_value = {
                "dense": base64.b64encode(dense.astype("float16")).decode("utf-8"),
                "dtype": "float16",
            }
        else:
            tile_value = {
                "dense": base64.b64encode(dense.astype("float32")).decode("utf-8"),
                "dtype": "float32",
            }

        generated_tiles += [(tile_id, tile_value)]
    return generated_tiles


def tileset_info(bwpath):
    """Get the tileset info for a bigWig file.

    Parameters
    ----------
    bwpath: string
        Path to the bigwig file

    Returns
    -------
    tileset_info: {
        'min_pos': [],
        'max_pos': [],
        'max_width': 131072
        'tile_size': 1024,
        'max_zoom': 7
    }
    """
    TILE_SIZE = 1024
    chromsizes = get_chromsizes(bwpath)
    min_tile_cover = np.ceil(np.sum(chromsizes) / TILE_SIZE)
    max_zoom = int(np.ceil(np.log2(min_tile_cover)))
    tileset_info = {
        "min_pos": [0],
        "max_pos": [TILE_SIZE * 2 ** max_zoom],
        "max_width": TILE_SIZE * 2 ** max_zoom,
        "tile_size": TILE_SIZE,
        "max_zoom": max_zoom,
    }
    return tileset_info


def chunk(
    bigwig: str,
    window_size: int,
    resolution: int,
    step_size: int,
    chroms: list,
    normalize: bool = True,
    percentile: float = 99.9,
    verbose: bool = False,
    print_per_chrom: callable = None,
):
    chrom_sizes = bbi.chromsizes(bigwig)
    base_bins = np.ceil(window_size / resolution).astype(int)

    num_total_windows = 0
    bins = np.ceil(window_size / resolution).astype(int)

    for chrom in chroms:
        chrom_size = chrom_sizes[chrom]
        num_total_windows += (
            np.ceil((chrom_size - window_size) / step_size).astype(int) + 1
        )

    values = np.zeros((num_total_windows, base_bins))

    start = 0
    for chrom in chroms:
        if chrom not in chrom_sizes:
            print("Skipping chrom (not in bigWig file):", chrom, chrom_sizes[chrom])
            continue

        chrom_size = chrom_sizes[chrom]
        num_windows = np.ceil((chrom_size - window_size) / step_size).astype(int) + 1

        start_bps = np.arange(0, chrom_size - window_size + step_size, step_size)
        end_bps = np.arange(window_size, chrom_size + step_size, step_size)

        end = start + num_windows

        values[start:end] = bbi.stackup(
            bigwig,
            [chrom] * start_bps.size,
            start_bps,
            end_bps,
            bins=bins,
            missing=0.0,
            oob=0.0,
        )

        if normalize:
            values[start:end] = utils.normalize(
                values[start:end], percentile=percentile
            )

        if verbose and not print_per_chrom:
            print(
                "Extracted",
                "{} windows".format(num_windows),
                "from {}".format(chrom),
                "with a max value of {}.".format(np.nanmax(values[start:end])),
            )

        if print_per_chrom:
            print_per_chrom()

        start = end

    return values


def get(
    bw_path: str, chrom: str, start: int, end: int, bins: int, missing: float = 0.0
):
    return bbi.fetch(bw_path, chrom, start, end, bins=bins, missing=missing)
