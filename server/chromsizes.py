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

SUPPORTED_CHROMOSOMES = [
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
    "chrY",
    "chrM",
]

hg19 = {
    "chr1": {"size": 249250621},
    "chr2": {"size": 243199373},
    "chr3": {"size": 198022430},
    "chr4": {"size": 191154276},
    "chr5": {"size": 180915260},
    "chr6": {"size": 171115067},
    "chr7": {"size": 159138663},
    "chr8": {"size": 146364022},
    "chr9": {"size": 141213431},
    "chr10": {"size": 135534747},
    "chr11": {"size": 135006516},
    "chr12": {"size": 133851895},
    "chr13": {"size": 115169878},
    "chr14": {"size": 107349540},
    "chr15": {"size": 102531392},
    "chr16": {"size": 90354753},
    "chr17": {"size": 81195210},
    "chr18": {"size": 78077248},
    "chr19": {"size": 59128983},
    "chr20": {"size": 63025520},
    "chr21": {"size": 48129895},
    "chr22": {"size": 51304566},
    "chrX": {"size": 155270560},
    "chrY": {"size": 59373566},
    "chrM": {"size": 16571},
}

grch38 = {
    "chr1": {"size": 248956422},
    "chr2": {"size": 242193529},
    "chr3": {"size": 198295559},
    "chr4": {"size": 190214555},
    "chr5": {"size": 181538259},
    "chr6": {"size": 170805979},
    "chr7": {"size": 159345973},
    "chr8": {"size": 145138636},
    "chr9": {"size": 138394717},
    "chr10": {"size": 133797422},
    "chr11": {"size": 135086622},
    "chr12": {"size": 133275309},
    "chr13": {"size": 114364328},
    "chr14": {"size": 107043718},
    "chr15": {"size": 101991189},
    "chr16": {"size": 90338345},
    "chr17": {"size": 83257441},
    "chr18": {"size": 80373285},
    "chr19": {"size": 58617616},
    "chr20": {"size": 64444167},
    "chr21": {"size": 46709983},
    "chr22": {"size": 50818468},
    "chrX": {"size": 156040895},
    "chrY": {"size": 57227415},
    "chrM": {"size": 16569},
}

all = {"hg19": hg19, "grch37": hg19, "hg38": grch38, "grch38": grch38}


def equals(cs1, cs2):
    for chrom in SUPPORTED_CHROMOSOMES:
        if cs1[chrom] != cs2[chrom]:
            return False
    return True
