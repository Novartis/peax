#!/usr/bin/env python

import os
import pathlib
import sys

module_path = os.path.abspath(os.path.join("../experiments"))
if module_path not in sys.path:
    sys.path.append(module_path)

from download import download_file

pathlib.Path("models2").mkdir(parents=True, exist_ok=True)

download_dir = "models"

base_url = "https://zenodo.org/record/2609763/files/"

download_file(
    "{}chip_w-120000_r-1000.h5?download=1".format(base_url),
    "chip_w-120000_r-1000.h5",
    dir="models2"
)
download_file(
    "{}chip_w-12000_r-100.h5?download=1".format(base_url),
    "chip_w-12000_r-100.h5",
    dir="models2"
)
download_file(
    "{}chip_w-3000_r-25.h5?download=1".format(base_url),
    "chip_w-3000_r-25.h5",
    dir="models2"
)
download_file(
    "{}dnase_w-120000_r-1000.h5?download=1".format(base_url),
    "dnase_w-120000_r-1000.h5",
    dir="models2"
)
download_file(
    "{}dnase_w-12000_r-100.h5?download=1".format(base_url),
    "dnase_w-12000_r-100.h5",
    dir="models2"
)
download_file(
    "{}dnase_w-3000_r-25.h5?download=1".format(base_url),
    "dnase_w-3000_r-25.h5",
    dir="models2"
)
