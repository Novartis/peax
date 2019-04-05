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

from download import download_file, download_encode_file

pathlib.Path("models").mkdir(parents=True, exist_ok=True)

download_file(
    "https://zenodo.org/record/2609763/files/dnase_w-3000_r-25.h5?download=1",
    "dnase_w-3000_r-25.h5",
    dir="models"
)

pathlib.Path("data").mkdir(parents=True, exist_ok=True)

# GM12878 DNase-seq read-depth normalized signal
download_encode_file("ENCFF158GBQ.bigWig")
