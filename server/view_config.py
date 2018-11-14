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

import copy
import numpy as np
from server import bigwig, defaults


def pad_target(pos_from, pos_to, amount=0.1):
    size = pos_to - pos_from
    padding = size * amount
    region = [int(pos_from), pos_to]
    viewport = [pos_from - padding, pos_to + padding]
    return region, viewport


def build(
    datasets,
    config,
    search_info: dict = None,
    domain: list = None,
    incl_predictions: bool = False,
    incl_autoencodings: bool = False,
    default: bool = False,
    hide_label: bool = False,
) -> dict:
    view_config = copy.deepcopy(defaults.VIEW_CONFIG)

    determine_x_domain = False
    region = None
    viewport = None

    if search_info is not None:
        region, viewport = pad_target(
            search_info["target_from"], search_info["target_to"]
        )

    if domain is not None:
        region, viewport = pad_target(*domain)

    # Setup defaut tracks
    for track in defaults.TOP_TRACKS:
        real_track = copy.deepcopy(track)

        combined_track_config = copy.deepcopy(defaults.COMBINED_TRACK)
        combined_track_config["uid"] = real_track["uid"] + "-combined"

        anno_track_config = copy.deepcopy(defaults.ANNOTATION_TRACK)
        anno_track_config["uid"] = real_track["uid"] + "-annotation"

        if region is not None:
            anno_track_config["options"]["regions"].append(region)

        if default and real_track.get("type") == "horizontal-gene-annotations":
            real_track["height"] *= 2
            real_track["options"]["fontSize"] = 10
            real_track["options"]["geneAnnoHeight"] = 10
            real_track["options"]["geneLabelPosition"] = "outside"
            real_track["options"]["geneStrandSpacing"] = 3

        combined_track_config["height"] = real_track.get("height")

        combined_track_config["contents"].extend([anno_track_config, real_track])

        view_config["views"][0]["tracks"]["top"].append(combined_track_config)

    if incl_predictions:
        combined_track_config = copy.deepcopy(defaults.COMBINED_TRACK)
        combined_track_config["uid"] = "class-probs-combined"

        anno_track_config = copy.deepcopy(defaults.ANNOTATION_TRACK)
        anno_track_config["uid"] = "class-probs-annotation"

        if region is not None:
            anno_track_config["options"]["regions"].append(region)

        probs_heatmap = copy.deepcopy(defaults.CLASS_PROB_TRACK)
        uid = "s{}p".format(search_info["id"])
        probs_heatmap["tilesetUid"] = uid
        probs_heatmap["uid"] = uid

        combined_track_config["height"] = probs_heatmap.get("height")
        combined_track_config["contents"].extend([anno_track_config, probs_heatmap])

        view_config["views"][0]["tracks"]["top"].append(combined_track_config)

    for i, dataset in enumerate(datasets):
        # Determine x-domain limits
        if not determine_x_domain:
            chromsizes = bigwig.get_chromsizes(dataset.filepath)
            chromcumsizes = np.cumsum(chromsizes)
            min_x = np.inf
            max_x = 0
            for chrom in config.chroms:
                csz = chromsizes[chrom]
                ccsz = chromcumsizes[chrom]
                min_x = min(min_x, ccsz - csz)
                max_x = max(max_x, ccsz)

            view_config["views"][0]["initialXDomain"] = [int(min_x), int(max_x)]

            determine_x_domain = True

        if bigwig.is_bigwig(dataset.filepath, dataset.filetype):
            combined_track_config = copy.deepcopy(defaults.COMBINED_TRACK)
            combined_track_config["uid"] = dataset.id + "-combined"

            anno_track_config = copy.deepcopy(defaults.ANNOTATION_TRACK)
            anno_track_config["uid"] = dataset.id + "-annotation"

            if region is not None:
                anno_track_config["options"]["regions"].append(region)

            bw_track_config = copy.deepcopy(defaults.BIGWIG_TRACK)
            bw_track_config["tilesetUid"] = dataset.id
            bw_track_config["uid"] = dataset.id

            if incl_autoencodings:
                encodings_track_config = copy.deepcopy(defaults.ENCODINGS_TRACK)
                uid = "{}|ae".format(bw_track_config["uid"])
                encodings_track_config["tilesetUid"] = uid
                encodings_track_config["uid"] = uid
                combined_track_config["contents"].append(encodings_track_config)

            if dataset.height:
                bw_track_config["height"] = dataset.height

            if dataset.fill:
                bw_track_config["options"]["fillColor"] = dataset.fill

            if dataset.name:
                bw_track_config["options"]["name"] = dataset.name

            if hide_label:
                bw_track_config["options"]["labelPosition"] = "hidden"

            if default:
                bw_track_config["height"] *= 3

            combined_track_config["height"] = bw_track_config.get("height")
            combined_track_config["contents"].extend(
                [anno_track_config, bw_track_config]
            )

            if i == datasets.size() - 1:
                # Add the chrom labels to the last track
                combined_track_config["contents"].append(defaults.CHROM_TRACK)

            view_config["views"][0]["tracks"]["top"].append(combined_track_config)

            if viewport is not None:
                view_config["views"][0]["initialXDomain"] = viewport

    return view_config


def height(datasets, config, incl_predictions: bool = False):
    view_config = build(datasets, config)
    total_height = 0

    for track in view_config["views"][0]["tracks"]["top"]:
        total_height += track.get("height", 0)

    extra_height = defaults.CLASS_PROB_TRACK.get("height", 0)

    return total_height, total_height + extra_height
