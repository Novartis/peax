#!/usr/bin/env python

import argparse
import datetime
import io
import json
import os
import pandas as pd
import pathlib
import requests
import sys
import time

from urllib.parse import urlencode


def find(
    settings: dict,
    strict: bool = False,
    base: str = ".",
    clear: bool = False,
    verbose: bool = False,
):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")

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
    file_types = list(settings["encode_file_types"].values())
    data_types = settings["data_types"]

    assay_slim = assay_slims[assay_type]

    filename = os.path.join(
        base,
        "datasets-{}-{}-{}{}.json".format(
            coord_system.lower(),
            abbr[assay_type] if assay_type in abbr else assay_type,
            timestamp,
            "-strict" if strict else "",
        ),
    )

    if pathlib.Path(filename).is_file() and not clear:
        print("File already exists. Use `--clear` to overwrite it.")
        return filename

    # 1. Search for experiments matching the assay type
    if verbose:
        print("Search for experiments... ", end="", flush=True)

    params = {
        "type": "Experiment",
        "status": "released",
        "assay_slims": assay_slim,
        "replicates.library.biosample.donor.organism.scientific_name": "Homo+sapiens",
        "assay_title": assay_type,
        "limit": "all",
        "format": "json",
    }
    biosample_term_names = [
        # cell lines
        "K562",
        "MCF-7",
        "H9",
        "GM03348",
        # primary cell
        "T-helper 1 cell",
        "dermis blood vessel endothelial cell",
        "trophoblast cell",
        # tissue
        "heart",
        "lung",
        "retina",
        "kidney",
        "thyroid gland",
    ]
    params_strict = {
        "award.project": "ENCODE",
        "biosample_ontology.classification": ["cell+line", "tissue", "primary+cell"],
        "biosample_ontology.term_name": [
            term.replace(" ", "+") for term in biosample_term_names
        ],
    }
    if strict:
        params.update(params_strict)
    param_str = urlencode(params, doseq=True).replace("%2B", "+")
    url = "https://www.encodeproject.org/search/"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers, params=param_str).json()

    experiments = [x["accession"] for x in response["@graph"]]

    assert len(experiments) == response["total"], "Number of results should equal"

    if verbose:
        print("done!")

    if verbose:
        print("Found {} human DNase-seq experiments".format(len(experiments)))

    # 2. Find data files associated to the experiments
    if verbose:
        print("Download metadata... ", end="", flush=True)

    selection = {"type": "Experiment", "files.file_type": file_types}
    base_url = "https://www.encodeproject.org/metadata"
    file_type = "metadata.tsv"
    headers = {"Accept": "text/tsv", "Content-Type": "application/json"}
    selection_path = urlencode(selection, doseq=True).replace("%2B", "+")
    url = "/".join([base_url, selection_path, file_type])
    data = {"elements": ["/experiments/{}/".format(e) for e in experiments]}

    urlData = requests.get(url, headers=headers, json=data).content
    metaData = pd.read_csv(io.StringIO(urlData.decode("utf-8")), sep="\t")

    if verbose:
        print("done!")

    # 3. Filter data and arrange in tuples
    is_no_error = metaData["Audit ERROR"].isnull()
    is_no_warning = metaData["Audit WARNING"].isnull()
    is_compliant = metaData["Audit NOT_COMPLIANT"].isnull()
    is_coord_system = metaData["Assembly"] == coord_system
    is_released = metaData["File Status"] == "released"
    is_assay_type = metaData["Assay"] == assay_type

    is_basic = (
        is_coord_system
        & is_released
        & is_assay_type
        & is_no_error
        & is_no_warning
        & is_compliant
    )

    if verbose:
        print(
            "Removed {} experiments due to auditing errors".format(
                metaData.loc[
                    is_coord_system
                    & is_released
                    & is_assay_type
                    & (~is_no_error | ~is_no_warning | ~is_compliant)
                ]["Experiment accession"]
                .unique()
                .shape[0]
            )
        )

    is_data_type = {}
    for data_type in data_types:
        is_data_type[data_type] = metaData["Output type"] == data_types[data_type]

    datasets = {}
    use_only_one_bio_replicate = True

    if strict:
        one_of_type = {}
        for term in biosample_term_names:
            one_of_type[term] = 0
        print

    k = 0
    for exp in metaData.loc[is_basic]["Experiment accession"].unique():
        is_exp = is_basic & (metaData["Experiment accession"] == exp)

        if strict:
            biosample_term_name = metaData.loc[is_exp].iloc[0]["Biosample term name"]
            if one_of_type[biosample_term_name] > 0:
                continue

        for sample in metaData.loc[is_coord_system & is_released & is_exp][
            "Biological replicate(s)"
        ].unique():
            is_sample = is_exp & (metaData["Biological replicate(s)"] == sample)

            data_by_type = {}
            for data_type in data_types:
                data_by_type[data_type] = metaData.loc[
                    is_sample & is_data_type[data_type]
                ]

            try:
                if exp not in datasets:
                    datasets[exp] = {}
                else:
                    if use_only_one_bio_replicate:
                        continue

                datasets[exp][sample] = {}

                for data_type in data_types:
                    entry = data_by_type[data_type].iloc[0]
                    datasets[exp][sample][data_type] = entry["File accession"]
                    biosample_term_name = entry["Biosample term name"]

            except IndexError:
                k += 1
                if exp in datasets:
                    # Remove the key in case there exist another replicate which has all data types
                    del datasets[exp]

            if strict:
                one_of_type[biosample_term_name] += 1

    if verbose:
        dnum = [ds for exp in datasets for ds in datasets[exp]]
        print(
            "Found {} experiments comprising {} datasets".format(
                len(datasets.keys()), len(dnum)
            )
        )

    # Save to file
    with open(filename, "w") as f:
        json.dump(datasets, f, indent=2)

    print("Saved dataset accessions to {}".format(filename))

    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Downloader")
    parser.add_argument(
        "-s", "--settings", help="path to the settings file", default="settings.json"
    )
    parser.add_argument(
        "-x", "--strict", action="store_true", help="search for CNN search data only"
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clear previously found datasets"
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

    find(settings, strict=args.strict, clear=args.clear, verbose=args.verbose)
