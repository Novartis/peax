#!/usr/bin/env python

import argparse
import datetime
import io
import json
import pandas as pd
import requests
import sys
import time

from urllib.parse import urlencode


parser = argparse.ArgumentParser(description="Peax Downloader")
parser.add_argument(
    "-s", "--settings", help="path to the settings file", default="settings.json"
)
parser.add_argument(
    "-c", "--clear", action="store_true", help="clears previously collected datasets"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="turn on verbose logging"
)

args = parser.parse_args()

try:
    with open(args.settings, "r") as f:
        settings = json.load(f)
except FileNotFoundError:
    print("Please provide a settings file via `--settings`")
    sys.exit(2)

assay_slims = {
    "ChIP-seq": "DNA+binding",
    "DNase-seq": "DNA+accessibility",
    "MNase-seq": "DNA+accessibility",
    "ATAC-seq": "DNA+accessibility",
    "FAIRE-seq": "DNA+accessibility",
}

abbr = {
    "ChIP-seq": "chip",
    "DNase-seq": "dnase",
    "MNase-seq": "mnase",
    "ATAC-seq": "atac",
    "FAIRE-seq": "faire",
}

coord_system = settings["coord_system"]
assay_type = settings["assay_type"]
file_types = list(settings["encode_file_types"].keys())
data_types = settings["data_types"]

assay_slim = assay_slims[assay_type]

# 1. Search for experiments matching the assay type
if args.verbose:
    print("Search for experiments...", end="", flush=True)

params = {
    "type": "Experiment",
    "status": "released",
    "assay_slims": assay_slim,
    "replicates.library.biosample.donor.organism.scientific_name": "Homo+sapiens",
    "assay_title": assay_type,
    "limit": "all",
    "format": "json",
}
param_str = urlencode(params, doseq=True).replace("%2B", "+")
url = "https://www.encodeproject.org/search/"
headers = {"Accept": "application/json"}

response = requests.get(url, headers=headers, params=param_str).json()

experiments = [x["accession"] for x in response["@graph"]]

assert len(experiments) == response["total"], "Number of results should equal"

if args.verbose:
    print("done!")

if args.verbose:
    print("Found {} human DNase-seq experiments".format(len(experiments)))

# 2. Find data files associated to the experiments
if args.verbose:
    print("Download metadata...", end="", flush=True)

selection = {"type": "Experiment", "files.file_type": file_types}
base_url = "https://www.encodeproject.org/metadata"
file_type = "metadata.tsv"
headers = {"Accept": "text/tsv", "Content-Type": "application/json"}
selection_path = urlencode(selection, doseq=True).replace("%2B", "+")
url = "/".join([base_url, selection_path, file_type])
data = {"elements": ["/experiments/{}/".format(e) for e in experiments]}

urlData = requests.get(url, headers=headers, json=data).content
metaData = pd.read_csv(io.StringIO(urlData.decode("utf-8")), sep="\t")

if args.verbose:
    print("done!")

# 3. Filter data and arrange in tuples

is_coord_system = metaData["Assembly"] == coord_system
is_released = metaData["File Status"] == "released"
is_dnase_seq = metaData["Assay"] == assay_type

is_data_type = {}
for data_type in data_types:
    is_data_type[data_type] = metaData["Output type"] == data_types[data_type]

is_rdn_signal = metaData["Output type"] == "read-depth normalized signal"
is_narrow_peaks = metaData["Output type"] == "peaks"
is_broad_peaks = metaData["Output type"] == "hotspots"

datasets = {}
use_only_one_bio_replicate = True

k = 0
for exp in metaData.loc[is_coord_system & is_released & is_dnase_seq][
    "Experiment accession"
].unique():
    is_exp = (
        is_coord_system
        & is_released
        & is_dnase_seq
        & (metaData["Experiment accession"] == exp)
    )

    for sample in metaData.loc[is_coord_system & is_released & is_exp][
        "Biological replicate(s)"
    ].unique():
        is_sample = is_exp & (metaData["Biological replicate(s)"] == sample)

        data_rdn_signal = metaData.loc[is_sample & is_rdn_signal]
        data_narrow_peaks = metaData.loc[is_sample & is_narrow_peaks]
        data_broad_peaks = metaData.loc[is_sample & is_broad_peaks]

        data_by_type = {}
        for data_type in data_types:
            data_by_type[data_type] = metaData.loc[is_sample & is_data_type[data_type]]

        try:
            if exp not in datasets:
                datasets[exp] = {}
            else:
                if use_only_one_bio_replicate:
                    continue

            datasets[exp][sample] = {}

            for data_type in data_types:
                datasets[exp][sample][data_type] = data_by_type[data_type].iloc[0][
                    "File accession"
                ]

        except IndexError:
            k += 1
            if exp in datasets:
                # Remove the key in case there exist another replicate which has all data types
                del datasets[exp]

if args.verbose:
    dnum = [ds for exp in datasets for ds in datasets[exp]]
    print(
        "Found {} experiments comprising {} datasets".format(
            len(datasets.keys()), len(dnum)
        )
    )

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")

# Save to file
filename = "datasets-{}-{}-{}.json".format(
    coord_system.lower(),
    abbr[assay_type] if assay_type in abbr else assay_type,
    timestamp,
)
with open(filename, "w") as f:
    json.dump(datasets, f, indent=2)

print("Saved dataset accessions to {}".format(filename))
