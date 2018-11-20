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
import numpy as np
from typing import Callable, List

from server import bigwig, utils

Vector = List[float]

TILE_SIZE = 1024

TILESET_INFO = {
    "filetype": "none",
    "datatype": "vector",
    "coordSystem": "hg19",
    "coordSystem2": "hg19",
}


def get_values(
    v: Vector,
    v_res: int,
    v_len_abs: int,
    v_offset_abs: int,
    offset: int,
    start: int,
    end: int,
    bins: int,
    res: int,
    missing: float = np.nan,
    aggregator: Callable = np.mean,
    scaleup_aggregator: Callable = np.mean,
) -> Vector:
    abs_start = offset + start
    abs_end = offset + end

    rel_start = max(abs_start, v_offset_abs)
    rel_end = min(abs_end, v_offset_abs + v_len_abs)

    out = np.zeros(bins)
    out[:] = missing

    # Imaging:
    # - RS = relative start
    # - RE = relative end
    # - AS = absolute start
    # - AE = absolute end
    # - ____ region to be cut out
    # - **** availabe data
    # |---RS___RE---AS******AE---RS___RE---|
    if rel_start >= abs_end or rel_end <= abs_start:
        # interval is outside
        return out

    else:
        # Either that start or the end lies within the available data vector
        start_res = np.floor((rel_start - v_offset_abs) / v_res).astype(int)
        end_res = np.ceil((rel_end - v_offset_abs) / v_res).astype(int)

        res_ratio = v_res / res

        data = np.zeros(bins)
        data[:] = missing

        if res_ratio < 1:
            # output resolution is greater than the data resolution, hence,
            # we need to aggregate the data
            data = utils.zoom_array(
                v[start_res:end_res], (bins,), aggregator=aggregator
            )

        elif res_ratio > 1:
            data = utils.scaleup_vector(
                v[start_res:end_res], bins, aggregator=scaleup_aggregator
            )

        if rel_end > abs_end:
            # Intervals overlaps with the start
            # ^ = Start; $ = end
            # Tile request: ----------^======$----
            # Data        : -------^======$-------
            out[out.size - data.size :] = data

        elif rel_start < abs_start:
            # Intervals overlaps with the end
            # ^ = Start; $ = end;
            # Tile request: ----^======$----------
            # Data        : -------^======$-------
            out[: data.size] = data

        else:
            out = data

        return out


def get_tile(
    v: np.ndarray,
    v_res: int,
    v_len_abs: int,
    v_offset_abs: int,
    zoom_level: int,
    start_pos: int,
    end_pos: int,
    chrom_sizes,
    chrom_offsets,
    aggregator=np.mean,
    scaleup_aggregator=np.mean,
):
    resolutions = bigwig.get_zoom_resolutions(chrom_sizes)
    resolution = resolutions[zoom_level]  # Number of bp per bin

    arrays = []
    for cid, start, end in bigwig.abs2chr(chrom_sizes, start_pos, end_pos):
        n_bins = int(np.ceil((end - start) / resolution))
        try:
            offset = chrom_offsets.values[cid]
            clen = chrom_sizes.values[cid]

            x = get_values(
                v,
                v_res,
                v_len_abs,
                v_offset_abs,
                offset,
                start,
                end,
                n_bins,
                resolution,
                aggregator=aggregator,
                scaleup_aggregator=scaleup_aggregator,
            )

            # drop the very last bin if it is smaller than the resolution
            if end == clen and clen % resolution != 0:
                x = x[:-1]

        except IndexError:
            # beyond the range of the available chromosomes
            # probably means we've requested a range of absolute
            # coordinates that stretch beyond the end of the genome
            x = np.zeros(n_bins)

        arrays.append(x)

    return np.concatenate(arrays)


def tiles(
    v: np.ndarray,
    v_res: int,
    v_len_abs: int,
    v_offset_abs: int,
    tile_ids,
    chrom_sizes,
    aggregator=np.mean,
    scaleup_aggregator=np.mean,
):
    """[summary].

    [description]

    Args:
        v: np.ndarray: [description]
        v_res: int: [description]
        v_len_abs: int: [description]
        v_offset_abs: int: [description]
        tile_ids: [description]
        chrom_sizes: [description]

    Returns:
        [description]
        [type]
    """

    generated_tiles = []
    for tile_id in tile_ids:
        tile_id_parts = tile_id.split(".")
        tile_position = list(map(int, tile_id_parts[1:3]))
        zoom_level = tile_position[0]
        tile_pos = tile_position[1]

        # this doesn't combine multiple consequetive ids, which
        # would speed things up
        chrom_offsets = np.cumsum(chrom_sizes) - chrom_sizes
        max_depth = bigwig.get_quadtree_depth(chrom_sizes)
        tile_size = TILE_SIZE * 2 ** (max_depth - zoom_level)
        start_pos = tile_pos * tile_size
        end_pos = start_pos + tile_size

        dense = get_tile(
            v,
            v_res,
            v_len_abs,
            v_offset_abs,
            zoom_level,
            start_pos,
            end_pos,
            chrom_sizes,
            chrom_offsets,
            aggregator=aggregator,
            scaleup_aggregator=scaleup_aggregator,
        )

        dense[np.isnan(dense)] = 0.0

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


def tileset_info(chromsizes, resolution):
    """Get the tileset info for a bigWig file.

    Parameters
    ----------
    chromsizes: pd.Dataframe
        Chromsizes dataframe

    resolution:

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
    min_tile_cover = np.ceil(sum(chromsizes) / TILE_SIZE)
    step_max_zoom = int(np.floor(np.log2(resolution)))
    max_zoom = int(np.ceil(np.log2(min_tile_cover)))
    tileset_info = {
        "min_pos": [0],
        "max_pos": [TILE_SIZE * 2 ** max_zoom],
        "max_width": TILE_SIZE * 2 ** max_zoom,
        "tile_size": TILE_SIZE,
        "max_zoom": max_zoom - step_max_zoom,
    }
    return tileset_info
