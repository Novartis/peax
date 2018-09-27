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

import wget
from pathlib import Path

Path("data").mkdir(parents=True, exist_ok=True)

# GM12878 DNase-seq read-depth normalized signal
bw = "data/ENCFF158GBQ.bigWig"
if not Path(bw).is_file():
    print("Download data...")
    wget.download(
        "https://www.encodeproject.org/files/ENCFF158GBQ/@@download/ENCFF158GBQ.bigWig",
        "data/ENCFF158GBQ.bigWig",
    )
    print("Done!")
