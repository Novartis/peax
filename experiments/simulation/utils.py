import higlass as hg
import numpy as np
import pandas as pd

from clodius.tiles.format import format_dense_tile


def create_bed_vector_tileset(
    bedlike_filepath: str,
    chromsizes_filepath: str = None,
    uuid: str = None,
    aggregator: callable = np.mean,
    log_scale: bool = False,
    categories: dict = None,
):
    TILE_SIZE = 1024
    chromsizes = pd.read_csv(
        chromsizes_filepath,
        sep="\t",
        index_col=0,
        usecols=[0, 1],
        names=[None, "size"],
        header=None,
    )
    cum_chromsizes = np.cumsum(chromsizes.values)
    min_tile_cover = np.ceil(np.sum(chromsizes) / TILE_SIZE)
    max_zoom = int(np.ceil(np.log2(min_tile_cover)))
    resolutions = [2 ** x for x in range(max_zoom + 1)][::-1]

    bedlike = pd.read_csv(
        bedlike_filepath,
        sep="\t",
        index_col=None,
        usecols=[0, 1, 2, 3, 4],
        names=["chrom", "start", "end", "name", "score"],
        header=None,
    )

    dense = np.zeros(cum_chromsizes[-1])

    # Densify bed data for later downsampling
    k = 0
    if categories is None:
        for region in bedlike.iterrows():
            length = int(region[1]["end"] - region[1]["start"])
            dense[k : k + length] = region[1]["score"]
            k += length
    else:
        for region in bedlike.iterrows():
            length = int(region[1]["end"] - region[1]["start"])
            try:
                dense[k : k + length] = categories[region[1]["name"]]
            except KeyError:
                dense[k : k + length] = categories["__others__"]
            k += length

    if log_scale:
        dense += 1
        dense = np.log(dense)

    def tileset_info(chromsizes):
        tileset_info = {
            "min_pos": [0],
            "max_pos": [TILE_SIZE * 2 ** max_zoom],
            "max_width": TILE_SIZE * 2 ** max_zoom,
            "tile_size": TILE_SIZE,
            "max_zoom": max_zoom,
        }
        return tileset_info

    def abs2genomic(chromsizes, start_pos, end_pos):
        abs_chrom_offsets = np.r_[0, cum_chromsizes]
        cid_lo, cid_hi = (
            np.searchsorted(abs_chrom_offsets, [start_pos, end_pos], side="right") - 1
        )
        rel_pos_lo = start_pos - abs_chrom_offsets[cid_lo]
        rel_pos_hi = end_pos - abs_chrom_offsets[cid_hi]
        start = rel_pos_lo
        for cid in range(cid_lo, cid_hi):
            yield cid, start, int(chromsizes.iloc[cid])
            start = 0
        yield cid_hi, start, rel_pos_hi

    def downsample(data, bins):
        dim = data.shape[0]

        assert dim >= bins

        # Downsampling factor
        factor = np.round(dim / bins)

        # Temporary dimension to support downsampling by an integer
        tmp_dim = int(bins * factor)
        diff = tmp_dim - dim

        left_pad = int(np.floor(np.abs(diff) / 2))
        right_pad = int(np.ceil(np.abs(diff) / 2))

        tmp = np.zeros(tmp_dim)

        if diff == 0:
            tmp = data
        elif diff > 0:
            # tmp is larger than data
            tmp[left_pad : tmp_dim - right_pad] = data
            tmp[:left_pad] = data[0]
            tmp[-right_pad:] = data[-1]
        else:
            # tmp is smaller than data
            tmp[:] = data[left_pad : dim - right_pad]

        return aggregator(tmp.reshape((int(tmp_dim / factor), -1)), axis=1)

    def fetch(chrom, start, end, bins):
        # Downsample
        return downsample(dense[start:end], bins)

    def get_tile(zoom_level, start_pos, end_pos):
        binsize = resolutions[zoom_level]

        arrays = []
        for cid, start, end in abs2genomic(chromsizes, start_pos, end_pos):
            bins = int(np.ceil((end - start) / binsize))
            try:
                chrom = chromsizes.index[cid]
                clen = chromsizes.values[cid]

                x = fetch(chrom, start, end, bins)

                # drop the very last bin if it is smaller than the binsize
                if end == clen and clen % binsize != 0:
                    x = x[:-1]
            except IndexError as e:
                # beyond the range of the available chromosomes
                # probably means we've requested a range of absolute
                # coordinates that stretch beyond the end of the genome
                x = np.zeros(bins)

            arrays.append(x)

        return np.concatenate(arrays)

    def tiles(tile_ids):
        generated_tiles = []

        for tile_id in tile_ids:
            # decompose the tile zoom and location
            _, zoom_level, tile_pos = tile_id.split(".")
            zoom_level = int(zoom_level)
            tile_pos = int(tile_pos)

            tile_size = TILE_SIZE * 2 ** (max_zoom - zoom_level)
            start_pos = tile_pos * tile_size
            end_pos = start_pos + tile_size

            # generate the tile
            data = get_tile(zoom_level, start_pos, end_pos)

            # format the tile response
            generated_tiles.append((tile_id, format_dense_tile(data)))

        return generated_tiles

    return hg.Tileset(
        tileset_info=lambda: tileset_info(chromsizes),
        tiles=lambda tids: tiles(tids),
        uuid=uuid,
    )


def extract_annotations(bedfile, features, chromsizes_file):
    chromsizes = pd.read_csv(
        chromsizes_file,
        sep="\t",
        index_col=0,
        usecols=[0, 1],
        names=[None, "size"],
        header=None,
    )
    cum_chromsizes = chromsizes.cumsum() - chromsizes.iloc[0]["size"]

    num_annotations_type = 0
    for feature in features:
        for annotation_type in features[feature]:
            num_annotations_type = max(num_annotations_type, annotation_type)
    num_annotations_type += 1

    annotations = []
    for i in range(num_annotations_type):
        annotations.append([])

    bed = pd.read_csv(
        bedfile,
        sep="\t",
        index_col=None,
        usecols=[0, 1, 2, 3, 4],
        names=["chrom", "start", "end", "name", "score"],
        header=None,
    )

    for region in bed.iterrows():
        feature = region[1]["name"]
        if feature in features:
            for annotation_type in features[feature]:
                offset = cum_chromsizes.loc[region[1]["chrom"]]["size"]
                annotations[annotation_type].append(
                    [offset + region[1]["start"], offset + region[1]["end"]]
                )

    return annotations


def create_annotation_track(uid, regions):
    return hg.Track(
        "horizontal-1d-annotations",
        uid=uid,
        position="top",
        height=8,
        options={
            "trackBorderWidth": 1,
            "trackBorderColor": "#f2f2f2",
            "regions": regions,
            "minRectWidth": 4,
            "fill": "#c17da5",
            "fillOpacity": 1,
            "strokeWidth": 0,
        },
    )
