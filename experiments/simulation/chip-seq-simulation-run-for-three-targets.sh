#!/bin/bash

# budding yeast has 12 Mbp = 12000000
# c elegans fly has 100 Mbp = 100000000
LENGTH=12000000
NUM_READS=1000000

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
      -n|--num) # Number of reads
      NUM_READS="$2"
      shift # past argument
      shift # past value
      ;;
      -d|--distortion) # Run with distorted peaks
      DISTORTION="$2"
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

if [ "$DISTORTION" == "spiky" ] ; then
  POST_FIX="-spiky-peaks"
elif [ "$DISTORTION" == "distorted" ] ; then
  POST_FIX="-distorted-peaks"
else
  DISTORTION=false
fi

SUFFIX="$LENGTH-$NUM_READS"


if [ -f "data/simulated-genome-$SUFFIX.fa" ] && [ -f "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.fastq" ]  && [ -f "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.fastq" ] && [ -f "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.fastq" ] && [ -f "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.fastq" ] && [ -f "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.fastq" ] && [ -f "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.fastq" ] && [ "$CLEAR" != true ] ; then
  echo "Skip ChIP-seq read simulation"
else
  echo "Clear previous fastq files..."
  > "data/simulated-genome-$SUFFIX.fa"
  > "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.fastq"
  > "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.fastq"
  > "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.fastq"
  > "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.fastq"
  > "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.fastq"
  > "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.fastq"

  echo "Simulate ChIP-seq reads..."
  Rscript chip-seq-simulation-generate-reads-for-three-targets.R $LENGTH $NUM_READS $DISTORTION
fi

echo "Index the genome..."
bwa index "data/simulated-genome-$SUFFIX.fa"

echo "Align reads..."
bwa mem -t 2 "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.fastq" > "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.sam"
bwa mem -t 2 "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.fastq" > "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.sam"
bwa mem -t 2 "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.fastq" > "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.sam"
bwa mem -t 2 "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.fastq" > "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.sam"
bwa mem -t 2 "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.fastq" > "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.sam"
bwa mem -t 2 "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.fastq" > "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.sam"

echo "Convert .sam to .bam..."
samtools view -bT "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.sam" > "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.bam"
samtools view -bT "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.sam" > "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.bam"
samtools view -bT "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.sam" > "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.bam"
samtools view -bT "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.sam" > "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.bam"
samtools view -bT "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.sam" > "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.bam"
samtools view -bT "data/simulated-genome-$SUFFIX.fa" "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.sam" > "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.bam"

echo "Sort reads by position..."
samtools sort -O bam -T temp "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.bam" > "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.sorted.bam"
samtools sort -O bam -T temp "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.bam" > "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.sorted.bam"
samtools sort -O bam -T temp "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.bam" > "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.sorted.bam"
samtools sort -O bam -T temp "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.bam" > "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.sorted.bam"
samtools sort -O bam -T temp "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.bam" > "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.sorted.bam"
samtools sort -O bam -T temp "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.bam" > "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.sorted.bam"

echo "Index position-sorted reads..."
samtools index "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.sorted.bam"
samtools index "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.sorted.bam"
samtools index "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.sorted.bam"
samtools index "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.sorted.bam"
samtools index "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.sorted.bam"
samtools index "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.sorted.bam"

echo "Compute signal..."
bamCompare -b1 "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.sorted.bam" -b2 "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.sorted.bam" -o "data/fold-change-target-1$POST_FIX-$SUFFIX.bigWig"
bamCompare -b1 "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.sorted.bam" -b2 "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.sorted.bam" -o "data/fold-change-target-2$POST_FIX-$SUFFIX.bigWig"
bamCompare -b1 "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.sorted.bam" -b2 "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.sorted.bam" -o "data/fold-change-target-3$POST_FIX-$SUFFIX.bigWig"
bamCoverage -b "data/reads-3-targets-target-1$POST_FIX-$SUFFIX.sorted.bam" -o "data/signal-target-1$POST_FIX-$SUFFIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50
bamCoverage -b "data/reads-3-targets-target-1-background$POST_FIX-$SUFFIX.sorted.bam" -o "data/signal-target-1-background$POST_FIX-$SUFFIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50
bamCoverage -b "data/reads-3-targets-target-2$POST_FIX-$SUFFIX.sorted.bam" -o "data/signal-target-2$POST_FIX-$SUFFIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50
bamCoverage -b "data/reads-3-targets-target-2-background$POST_FIX-$SUFFIX.sorted.bam" -o "data/signal-target-2-background$POST_FIX-$SUFFIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50
bamCoverage -b "data/reads-3-targets-target-3$POST_FIX-$SUFFIX.sorted.bam" -o "data/signal-target-3$POST_FIX-$SUFFIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50
bamCoverage -b "data/reads-3-targets-target-3-background$POST_FIX-$SUFFIX.sorted.bam" -o "data/signal-target-3-background$POST_FIX-$SUFFIX.bigWig" --binSize 25 --normalizeUsing RPGC --effectiveGenomeSize $LENGTH --extendReads 50

echo "Convert feature bed to bigBed..."
bedToBigBed "data/simulated-features-3-targets-$SUFFIX.bed" "data/simulated-genome-chrom-sizes-$SUFFIX.tsv" "data/simulated-features-3-targets-$SUFFIX.bigBed"
