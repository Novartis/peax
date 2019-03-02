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

CACHE_DIR = "cache"

# If set to `False` the chunked, encoded, and potentially autoencoded data will not be
# cached. Unless you know what you're doing and you have rather small data leave
# caching on.
CACHING = True

DB_PATH = "search.db"

CHROMS = [
    "chr1",
    "chr2",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr14",
    "chr15",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chrX",
]

COORDS = "hg19"

STEP_FREQ = 2

MIN_CLASSIFICATIONS = 20

TILE_SIZE = 1024

COMBINED_TRACK = {"uid": "???", "type": "combined", "contents": []}

ANNOTATION_TRACK = {
    "uid": "???",
    "type": "horizontal-1d-annotations",
    "options": {
        "regions": [],
        "minRectWidth": 3,
        "fill": "#dcecf7",
        "fillOpacity": 0.2,
        "stroke": "#dcecf7",
        "strokeOpacity": 1.0,
        "strokePos": ["left", "right"],
        "strokeWidth": 2,
    },
}

SELECTION_TRACK = {
    "uid": "???",
    "type": "horizontal-1d-annotations",
    "options": {
        "regions": [],
        "minRectWidth": 3,
        "fill": "#e59f00",
        "fillOpacity": 1.0,
        "strokeWidth": 0,
    },
    "height": 6,
}

CLASS_PROB_TRACK = {
    "server": "//localhost:5000/api/v1",
    "tilesetUid": "s?p",
    "uid": "???",
    "type": "horizontal-1d-heatmap",
    "options": {
        "colorRange": [
            "#f2f2f2",
            "#f2f2f2",
            "#f2f2f2",
            "#f2f2f2",
            "#bac8e2",
            "#7494c5",
            "#0064a8",
        ],
        "labelPosition": "hidden",
        "showMousePosition": True,
        "mousePositionColor": "#000000",
        "valueScaleMin": 0,
        "valueScaleMax": 1,
        "showTooltip": True,
    },
    "height": 6,
}

BIGWIG_TRACK = {
    "server": "//localhost:5000/api/v1",
    "tilesetUid": "???",
    "uid": "???",
    "type": "horizontal-bar",
    "options": {
        "axisPositionHorizontal": "left",
        "axisMargin": 32,
        "barFillColor": "#00266d",
        "barOpacity": 0.75,
        "labelBackgroundOpacity": 0.0001,
        "labelColor": "black",
        "labelTextOpacity": 0.5,
        "labelPosition": "topLeft",
        "labelLeftMargin": 56,
        "trackBorderWidth": 0,
        "showMousePosition": True,
        "mousePositionColor": "#000000",
        "showTooltip": True,
    },
    "height": 60,
}

ENCODINGS_TRACK = {
    "server": "//localhost:5000/api/v1",
    "tilesetUid": "???",
    "uid": "???",
    "type": "horizontal-bar",
    "options": {
        "axisPositionHorizontal": "right",
        "barFillColor": "#CC168C",
        "barOpacity": 0.33,
        "labelPosition": "hidden",
        "trackBorderWidth": 0,
        "showMousePosition": True,
        "mousePositionColor": "#000000",
        "showTooltip": True,
        "valueScaleMin": 0,
        "valueScaleMax": 1,
    },
    "height": 60,
}

TOP_TRACKS = [
    {
        "server": "//higlass.io/api/v1",
        "tilesetUid": "OHJakQICQD6gTD7skx4EWA",
        "uid": "gene-annotations-hg19",
        "type": "horizontal-gene-annotations",
        "height": 24,
        "options": {
            "labelPosition": "hidden",
            "plusStrandColor": "black",
            "minusStrandColor": "black",
            "trackBorderWidth": 0,
            "showMousePosition": True,
            "mousePositionColor": "black",
            "fontSize": 8,
            "geneAnnoHeight": 8,
            "geneLabelPosition": "inside",
            "geneStrandSpacing": 2,
        },
    }
]

CHROM_TRACK = {
    "chromInfoPath": "//s3.amazonaws.com/pkerp/data/hg19/chromSizes.tsv",
    "uid": "chrom-labels-hg19",
    "type": "horizontal-chromosome-labels",
    "name": "Chromosome Labels (hg19)",
    "height": 10,
    "options": {
        "fontSize": 8,
        "fontIsLeftAligned": True,
        "showMousePosition": True,
        "mousePositionColor": "#000000",
    },
}

VIEW_CONFIG = {
    "editable": False,
    "zoomFixed": False,
    "trackSourceServers": ["//localhost:5000/api/v1", "//higlass.io/api/v1"],
    "exportViewUrl": "//localhost:5000/api/v1/viewconf/",
    "views": [
        {
            "initialXDomain": [0, 3095677412],
            "genomePositionSearchBoxVisible": False,
            "tracks": {"top": [], "left": [], "center": [], "right": [], "bottom": []},
            "layout": {
                "w": 12,
                "h": 12,
                "x": 0,
                "y": 0,
                "moved": False,
                "static": False,
            },
        }
    ],
}
