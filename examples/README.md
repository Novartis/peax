# Peax Examples

We provide several example configurations to give you a chance to start quickly.
All of the examples use [our trained convolutional autoencoder models](#autoencoders).

## Examples

We provide three simple and three more advanced examples that you can start
by calling `make <NAME>`.

| Name                        | Dtypes (Num Tracks)              | Window Size |
| --------------------------- | -------------------------------- | ----------- |
| example-3kb                 | DNase                            | 3 kb        |
| example-12kb                | DNase                            | 12 kb       |
| example-120kb               | DNase                            | 120 kb      |
| encode-e11-5-limb           | DNase, H3K27ac                   | 3 kb        |
| encode-e11-5-face-hindbrian | DNase                            | 3 kb        |
| roadmap-e116-gm12878        | DNase, H3K27ac, H3K4me1, H3K4me3 | 12 kb       |

Each example consists of a python script for downloading the approriate datasets
(`download-<NAME>.py`) and the accompanying configuration (`config-<NAME>.json`).

## Autoencoders

We have trained 6 autoencoder on DNase-seq and histone mark ChIP-seq datasets.
The DNase-seq autoencoders were trained on 120 datasets from ENCODE and the
ChIP-seq autoencoders were trained on 343 datasets each targeting one of the
following histone marks:

  - H3K4me1
  - H3K4me3
  - H3K27ac
  - H3K9ac
  - H3K27me3
  - H3K9me3
  - H3K36me3

For each data type we trained 3 autoencoders on the following window sizes and
binning.

| Dtype   | Window Size | Binning |
| ------- | ----------- | ------- |
| ChIP    | 3 kb        | 25 bp   |
| ChIP    | 12 kb       | 100 bp  |
| ChIP    | 120 kb      | 1000 bp |
| DNase   | 3 kb        | 25 bp   |
| DNase   | 12 kb       | 100 bp  |
| DNase   | 120 kb      | 1000 bp |

**Download:** https://zenodo.org/record/2609763

Please see our preprint for a detailed description of the training and parameter settings:

> Lekschas et al., 2019, [Peax: Interactive Visual Pattern Search in Sequential Data Using Unsupervised Deep Representation Learning](https://www.biorxiv.org/content/10.1101/597518v1)
