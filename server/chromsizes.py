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

import cooler
import pandas as pd
from server import bigwig

SUPPORTED_CHROMOSOMES_HUMAN = [
    "chr1",
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
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrX",
    "chrM",
]

SUPPORTED_CHROMOSOMES_MOUSE = [
    "chr1",
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
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chrX",
    "chrM",
]

SUPPORTED_CHROMOSOMES = {
    "hg19": SUPPORTED_CHROMOSOMES_HUMAN,
    "grch38": SUPPORTED_CHROMOSOMES_HUMAN,
    "mm9": SUPPORTED_CHROMOSOMES_MOUSE,
    "mm10": SUPPORTED_CHROMOSOMES_MOUSE,
}

hg19 = {
    "chr1": 249250621,
    "chr2": 243199373,
    "chr3": 198022430,
    "chr4": 191154276,
    "chr5": 180915260,
    "chr6": 171115067,
    "chr7": 159138663,
    "chr8": 146364022,
    "chr9": 141213431,
    "chr10": 135534747,
    "chr11": 135006516,
    "chr12": 133851895,
    "chr13": 115169878,
    "chr14": 107349540,
    "chr15": 102531392,
    "chr16": 90354753,
    "chr17": 81195210,
    "chr18": 78077248,
    "chr19": 59128983,
    "chr20": 63025520,
    "chr21": 48129895,
    "chr22": 51304566,
    "chrX": 155270560,
    "chrY": 59373566,
    "chrM": 16571,
}

grch38 = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
    "chrM": 16569,
}

mm9 = {
    "chr1": 197195432,
    "chr2": 181748087,
    "chr3": 159599783,
    "chr4": 155630120,
    "chr5": 152537259,
    "chr6": 149517037,
    "chr7": 152524553,
    "chr8": 131738871,
    "chr9": 124076172,
    "chr10": 129993255,
    "chr11": 121843856,
    "chr12": 121257530,
    "chr13": 120284312,
    "chr14": 125194864,
    "chr15": 103494974,
    "chr16": 98319150,
    "chr17": 95272651,
    "chr18": 90772031,
    "chr19": 61342430,
    "chrX": 166650296,
    "chrY": 15902555,
    "chrM": 16299,
}

mm10 = {
    "chr1": 195471971,
    "chr2": 182113224,
    "chr3": 160039680,
    "chr4": 156508116,
    "chr5": 151834684,
    "chr6": 149736546,
    "chr7": 145441459,
    "chr8": 129401213,
    "chr9": 124595110,
    "chr10": 130694993,
    "chr11": 122082543,
    "chr12": 120129022,
    "chr13": 120421639,
    "chr14": 124902244,
    "chr15": 104043685,
    "chr16": 98207768,
    "chr17": 94987271,
    "chr18": 90702639,
    "chr19": 61431566,
    "chrX": 171031299,
    "chrY": 91744698,
    "chrM": 16299,
}

all = {
    "hg19": hg19,
    "grch37": hg19,
    "hg38": grch38,
    "grch38": grch38,
    "mm9": mm9,
    "mm10": mm10,
}


def get(coords, filepath: str = None):
    if coords in all:
        return pd.Series(all[coords])[cooler.util.natsorted(all[coords].keys())]

    elif filepath is not None:
        return bigwig.get_chromsizes(filepath)

    raise ValueError("No chromsizes available")


def equals(cs1, cs2, coords, custom_chromosomes=None):
    chroms = (
        custom_chromosomes
        if custom_chromosomes is not None
        else SUPPORTED_CHROMOSOMES[coords]
    )
    for chrom in chroms:
        if cs1[chrom] != cs2[chrom]:
            return False
    return True
