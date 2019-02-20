#!/bin/bash

LENGTH=12000000

POSITIONAL=()
while [[ $# -gt 0 ]]
  do
  key="$1"

  case $key in
      -l|--length) # Length of the genome to be simulated
      LENGTH="$2"
      shift # past argument
      shift # past value
      ;;
      -s|--spiky-peaks) # Run with spiked peaks, i.e., punctuated increase of binding prob
      SPIKY=true
      POST_FIX="-spiky-peaks"
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

if [ -f data/simulated-genome.fa ] && [ -f "data/simulated-reads-chip$POST_FIX.fastq" ] && [ -f "data/simulated-reads-input$POST_FIX.fastq" ] && [ "$CLEAR" != true ] ; then
  echo "Skip ChIP-seq read simulation"
else
  echo "Simulate ChIP-seq reads..."
  Rscript chip-seq-simulation-generate-reads.R $LENGTH $SPIKY
fi

echo "Index the genome..."
bwa index data/simulated-genome.fa

echo "Align reads..."
bwa mem -t 2 data/simulated-genome.fa "data/simulated-reads-chip$POST_FIX.fastq" > "data/simulated-reads-chip$POST_FIX.sam"
bwa mem -t 2 data/simulated-genome.fa "data/simulated-reads-input$POST_FIX.fastq" > "data/simulated-reads-input$POST_FIX.sam"

echo "Convert .sam to .bam..."
samtools view -bT data/simulated-genome.fa "data/simulated-reads-chip$POST_FIX.sam" > "data/simulated-reads-chip$POST_FIX.bam"
samtools view -bT data/simulated-genome.fa "data/simulated-reads-input$POST_FIX.sam" > "data/simulated-reads-input$POST_FIX.bam"

echo "Sort reads by position..."
samtools sort -O bam -T temp "data/simulated-reads-chip$POST_FIX.bam" > "data/simulated-reads-chip$POST_FIX.sorted.bam"
samtools sort -O bam -T temp "data/simulated-reads-input$POST_FIX.bam" > "data/simulated-reads-input$POST_FIX.sorted.bam"

echo "Index position-sorted reads..."
samtools index "data/simulated-reads-chip$POST_FIX.sorted.bam"
samtools index "data/simulated-reads-input$POST_FIX.sorted.bam"

echo "Compute signal..."
bamCompare -b1 "data/simulated-reads-chip$POST_FIX.sorted.bam" -b2 "data/simulated-reads-input$POST_FIX.sorted.bam" -o "data/simulated-fold-change-signal$POST_FIX.bigWig"
bamCoverage -b "data/simulated-reads-chip$POST_FIX.sorted.bam" -o "data/simulated-chip-signal$POST_FIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50
bamCoverage -b "data/simulated-reads-input$POST_FIX.sorted.bam" -o "data/simulated-input-signal$POST_FIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50

echo "Convert feature bed to bigBed..."
bedToBigBed data/simulated-features.bed data/simulated-genome-chrom-sizes.tsv data/simulated-features.bigBed
