#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -l|--length)
    LENGTH="$2"
    shift # past argument
    shift # past value
    ;;
    -c|--clear)
    CLEAR=true
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ -f data/simulated-genome.fa ] && [ -f data/simulated-reads-chip.fastq ] && [ -f data/simulated-reads-input.fastq ] && [ "$CLEAR" != true ] ; then
  echo "Skip ChIP-seq read simulation"
else
  echo "Simulate ChIP-seq reads..."
  Rscript chip-seq-simulation-generate-reads.R $LENGTH
fi

echo "Index the genome..."
bwa index data/simulated-genome.fa

echo "Align reads..."
bwa mem -t 2 data/simulated-genome.fa data/simulated-reads-chip.fastq > data/simulated-reads-chip.sam
bwa mem -t 2 data/simulated-genome.fa data/simulated-reads-input.fastq > data/simulated-reads-input.sam

echo "Convert .sam to .bam..."
samtools view -bT data/simulated-genome.fa data/simulated-reads-chip.sam > data/simulated-reads-chip.bam
samtools view -bT data/simulated-genome.fa data/simulated-reads-input.sam > data/simulated-reads-input.bam

echo "Sort reads by position..."
samtools sort -O bam -T temp data/simulated-reads-chip.bam > data/simulated-reads-chip.sorted.bam
samtools sort -O bam -T temp data/simulated-reads-input.bam > data/simulated-reads-input.sorted.bam

echo "Index position-sorted reads..."
samtools index data/simulated-reads-chip.sorted.bam
samtools index data/simulated-reads-input.sorted.bam

echo "Compute genome-wide fold-change coverage..."
bamCompare -b1 data/simulated-reads-chip.sorted.bam -b2 data/simulated-reads-input.sorted.bam -o data/simulated-fold-change-signal.bigWig

echo "Convert feature bed to bigBed..."
bedToBigBed data/simulated-features.bed data/simulated-genome-chrom-sizes.tsv data/simulated-features.bigBed
