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


TILE_SIZE = 1024

TILESET_INFO = {
    "filetype": "bigbed",
    "datatype": "bedlike",
    "coordSystem": "hg19",
    "coordSystem2": "hg19",
}

FILE_EXT = {"bigbed", "bb"}


def is_bigwig(filepath=None, filetype=None):
    if filetype == "bigbed":
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
    """
    TODO: replace this with negspy

    Also, return NaNs from any missing chromosomes in bbi.fetch

    """
    chromsizes = bbi.chromsizes(bwpath)
    chromosomes = cooler.util.natsorted(chromsizes.keys())
    return pd.Series(chromsizes)[chromosomes]


def abs2genomic(chromsizes, start_pos, end_pos):
    abs_chrom_offsets = np.r_[0, np.cumsum(chromsizes.values)]
    cid_lo, cid_hi = (
        np.searchsorted(abs_chrom_offsets, [start_pos, end_pos], side="right")
        - 1
    )
    rel_pos_lo = start_pos - abs_chrom_offsets[cid_lo]
    rel_pos_hi = end_pos - abs_chrom_offsets[cid_hi]
    start = rel_pos_lo
    for cid in range(cid_lo, cid_hi):
        yield cid, start, chromsizes[cid]
        start = 0
    yield cid_hi, start, rel_pos_hi


def get_tile(bwpath, zoom_level, start_pos, end_pos):
    chromsizes = get_chromsizes(bwpath)
    resolutions = get_zoom_resolutions(chromsizes)
    binsize = resolutions[zoom_level]

    arrays = []
    for cid, start, end in abs2genomic(chromsizes, start_pos, end_pos):
        n_bins = int(np.ceil((end - start) / binsize))
        try:
            chrom = chromsizes.index[cid]
            clen = chromsizes.values[cid]

            x = bbi.fetch(
                bwpath, chrom, start, end, bins=n_bins, missing=np.nan
            )

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


def get_tile_by_id(bwpath, zoom_level, tile_pos):
    """
    Get the data for a bigWig tile given a tile id.
    Parameters
    ----------
    bwpath: string
        The path to the bigWig file (can be remote)
    zoom_level: int
        The zoom level to get the data for
    tile_pos: int
        The position of the tile
    """
    max_depth = get_quadtree_depth(get_chromsizes(bwpath))
    tile_size = TILE_SIZE * 2 ** (max_depth - zoom_level)

    start_pos = tile_pos * tile_size
    end_pos = start_pos + tile_size

    return get_tile(bwpath, zoom_level, start_pos, end_pos)


def tiles(bwpath, tile_ids):
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
    TILE_SIZE = 1024
    generated_tiles = []
    for tile_id in tile_ids:
        tile_id_parts = tile_id.split(".")
        tile_position = list(map(int, tile_id_parts[1:3]))
        zoom_level = tile_position[0]
        tile_pos = tile_position[1]

        # this doesn't combine multiple consequetive ids, which
        # would speed things up
        max_depth = get_quadtree_depth(get_chromsizes(bwpath))
        tile_size = TILE_SIZE * 2 ** (max_depth - zoom_level)
        start_pos = tile_pos * tile_size
        end_pos = start_pos + tile_size
        dense = get_tile(bwpath, zoom_level, start_pos, end_pos)

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
                "dense": base64.b64encode(dense.astype("float16")).decode(
                    "utf-8"
                ),
                "dtype": "float16",
            }
        else:
            tile_value = {
                "dense": base64.b64encode(dense.astype("float32")).decode(
                    "utf-8"
                ),
                "dtype": "float32",
            }

        generated_tiles += [(tile_id, tile_value)]
    return generated_tiles


def tileset_info(bwpath):
    """Get the tileset info for a bigWig file

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
    chromsizes = bbi.chromsizes(bwpath)
    chromosomes = cooler.util.natsorted(chromsizes.keys())
    chromsizes = pd.Series(chromsizes)[chromosomes]
    min_tile_cover = np.ceil(sum(chromsizes) / TILE_SIZE)
    max_zoom = int(np.ceil(np.log2(min_tile_cover)))
    tileset_info = {
        "min_pos": [0],
        "max_pos": [TILE_SIZE * 2 ** max_zoom],
        "max_width": TILE_SIZE * 2 ** max_zoom,
        "tile_size": TILE_SIZE,
        "max_zoom": max_zoom,
    }
    return tileset_info
