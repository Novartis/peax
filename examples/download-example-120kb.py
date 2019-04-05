#!/usr/bin/env python

import os
import pathlib
import sys

module_path = os.path.abspath(os.path.join("../experiments"))
if module_path not in sys.path:
    sys.path.append(module_path)

from download import download_file, download_encode_file

pathlib.Path("models").mkdir(parents=True, exist_ok=True)

download_file(
    "https://zenodo.org/record/2609763/files/dnase_w-120000_r-1000.h5?download=1",
    "dnase_w-120000_r-1000.h5",
    dir="models"
)

pathlib.Path("data").mkdir(parents=True, exist_ok=True)

# GM12878 DNase-seq read-depth normalized signal
download_encode_file("ENCFF158GBQ.bigWig")
