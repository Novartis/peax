#!/usr/bin/env python

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

import os
import pathlib
import sys

module_path = os.path.abspath(os.path.join("../experiments"))
if module_path not in sys.path:
    sys.path.append(module_path)

from download import download_file

pathlib.Path("models").mkdir(parents=True, exist_ok=True)

download_file(
    "https://zenodo.org/record/2609763/files/dnase_w-12000_r-100.h5?download=1",
    "dnase_w-12000_r-100.h5",
    dir="models"
)

download_file(
    "https://zenodo.org/record/2609763/files/chip_w-12000_r-100.h5?download=1",
    "chip_w-12000_r-100.h5",
    dir="models"
)

pathlib.Path("data").mkdir(parents=True, exist_ok=True)

download_dir = "data"

base_url = "https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/foldChange/"

# GM12878 DNase-seq read-depth normalized signal
download_file(base_url + "E116-DNase.fc.signal.bigwig", "E116-DNase.fc.signal.bigWig")

# GM12878 H3K4me1 ChIP-seq fold change over control
download_file(
    base_url + "E116-H3K4me1.fc.signal.bigwig", "E116-H3K4me1.fc.signal.bigWig"
)

# GM12878 H3K4me3 ChIP-seq fold change over control
download_file(
    base_url + "E116-H3K4me3.fc.signal.bigwig", "E116-H3K4me3.fc.signal.bigWig"
)

# GM12878 H3K27ac ChIP-seq fold change over control
download_file(
    base_url + "E116-H3K27ac.fc.signal.bigwig", "E116-H3K27ac.fc.signal.bigWig"
)
