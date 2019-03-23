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

from download import download_encode_file

pathlib.Path("data").mkdir(parents=True, exist_ok=True)

download_dir = "data"

# e11.5 midbrain DNase-seq read-depth normalized signal
download_encode_file("ENCFF906WCV.bigWig")

# e11.5 midbrain DNase-seq narrow peaks
download_encode_file("ENCFF606VGL.bigBed")

# e11.5 midbrain DNase-seq broad peaks
download_encode_file("ENCFF288OHT.bigBed")

# e11.5 midbrain ChIP-seq H3K27ac fold change over control
download_encode_file("ENCFF847OYS.bigWig")

# e11.5 midbrain ChIP-seq H3K27ac narrow peaks
download_encode_file("ENCFF762GGS.bigBed")

# e11.5 midbrain ChIP-seq H3K27ac broad peaks
download_encode_file("ENCFF004ZKP.bigBed")
