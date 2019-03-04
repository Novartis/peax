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

mm9 = {
    "chr1": {"size": 197195432},
    "chr2": {"size": 181748087},
    "chr3": {"size": 159599783},
    "chr4": {"size": 155630120},
    "chr5": {"size": 152537259},
    "chr6": {"size": 149517037},
    "chr7": {"size": 152524553},
    "chr8": {"size": 131738871},
    "chr9": {"size": 124076172},
    "chr10": {"size": 129993255},
    "chr11": {"size": 121843856},
    "chr12": {"size": 121257530},
    "chr13": {"size": 120284312},
    "chr14": {"size": 125194864},
    "chr15": {"size": 103494974},
    "chr16": {"size": 98319150},
    "chr17": {"size": 95272651},
    "chr18": {"size": 90772031},
    "chr19": {"size": 61342430},
    "chrX": {"size": 166650296},
    "chrY": {"size": 15902555},
    "chrM": {"size": 16299},
}

mm10 = {
    "chr1": {"size": 195471971},
    "chr2": {"size": 182113224},
    "chr3": {"size": 160039680},
    "chr4": {"size": 156508116},
    "chr5": {"size": 151834684},
    "chr6": {"size": 149736546},
    "chr7": {"size": 145441459},
    "chr8": {"size": 129401213},
    "chr9": {"size": 124595110},
    "chr10": {"size": 130694993},
    "chr11": {"size": 122082543},
    "chr12": {"size": 120129022},
    "chr13": {"size": 120421639},
    "chr14": {"size": 124902244},
    "chr15": {"size": 104043685},
    "chr16": {"size": 98207768},
    "chr17": {"size": 94987271},
    "chr18": {"size": 90702639},
    "chr19": {"size": 61431566},
    "chrX": {"size": 171031299},
    "chrY": {"size": 91744698},
    "chrM": {"size": 16299},
}

all = {
    "hg19": hg19,
    "grch37": hg19,
    "hg38": grch38,
    "grch38": grch38,
    "mm9": mm9,
    "mm10": mm10,
}


def equals(cs1, cs2):
    for chrom in SUPPORTED_CHROMOSOMES:
        if cs1[chrom] != cs2[chrom]:
            return False
    return True
