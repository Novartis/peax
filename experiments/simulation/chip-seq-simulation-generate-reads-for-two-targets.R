#!/usr/bin/env Rscript

# Avoid uninformative output by loading packages
sink("/dev/null")

# Install packages if not available
list.of.packages <- c("ChIPsim")
new.packages <- list.of.packages[
  !(list.of.packages %in% installed.packages()[,"Package"])
]
if(length(new.packages)) install.packages(
  new.packages, repos = "http://cran.us.r-project.org"
)

rm(list=ls())
capture.output(suppressMessages( require('ChIPsim') ))

# Turn the output back on
sink()

data_dir <- "data/"

#Parameters to play around with
#Transition probabilities
#Gamma distribution
#Enrichment over background
#Pareto distribution parameter
shapeA <- 1
scaleA <- 20
enrichmentA <- 5
shapeB <- 1
scaleB <- 18
enrichmentB <- 5
shapeAB <- 1
scaleAB <- 24
enrichmentAB <- 5
shapeBg <- 1
scaleBg <- 20
r1 <- 2
BindLength <- 50
MinLength <- 150
MaxLength <- 250
MeanLength <- 200
BindingAProb <- 0.006
BindingBProb <- 0.006
BindingABProb <- 0.006
BindingProb <- BindingAProb + BindingBProb + BindingABProb
BackgroundProb <- 1-BindingProb
BackgroundFeatureLength <- 1000
BindingFeatureLength <- 500
seed1 <- 1234
seed2 <- 1235
# number of reads
Nreads <- 1e5
# length of reads
LengthReads <- 50
# generate random genome
set.seed(seed1)

args = commandArgs(trailingOnly = TRUE)

# length of the chromosome (which is the same as the length of the genome 120 Mb)
chrLen <- c(1.2e7)
if (length(args) >= 1) chrLen = c(as.integer(args[[1]]))
if (length(args) >= 2) Nreads = c(as.integer(args[[2]]))
if (length(args) >= 3) {
  spiky = as.logical(args[[3]])
  postFix = "-spiky-peaks"
} else {
  spiky = FALSE
  postFix = ""
}

generateGenome <- function(chrLen) {
  chromosomes <- sapply(
    chrLen,
    function(n) paste(sample(c("A", "C", "G", "T"), n, replace = TRUE), collapse = "")
  )
  chrom_names <- paste("chr", seq_along(chromosomes), sep="")
  names(chromosomes) <- chrom_names
  genome <- DNAStringSet(chromosomes)
  chromosomes <- DNAString(chromosomes)

  # Save generated genome
  Biostrings::writeXStringSet(
    genome,
    file=paste(data_dir, "simulated-genome.fa", sep=""),
    "fasta",
    width=80,
    append=FALSE,
    compress=FALSE,
    compression_level=NA
  )
  # Choose a return value in case of error
  return(list(chrom_names=chrom_names, genome=genome, chromosomes=chromosomes))
}

getGenome <- function(chrLen) {
  out <- tryCatch(
    {
      genome <- readDNAStringSet("data/simulated-genome.fa")
      chrom_names <- paste("chr", seq_len(length(genome)), sep="")
      list(chrom_names=chrom_names, genome=genome, chromosomes=genome[[1]])
    },
    error=function(cond) {
      # Choose a return value in case of error
      return(generateGenome(chrLen))
    },
    warning=function(cond) {
      message(cond)
      # Choose a return value in case of warning
      return(list(genome=NULL, chromosomes=NULL))
    }
  )
  return(out)
}

out = getGenome(chrLen)

stopifnot(
  !is.null(out$chrom_names) && !is.null(out$genome) && !is.null(out$chromosomes)
)

if (length(out$chromosomes) != chrLen) out = generateGenome(chrLen)

chrom_names = out$chrom_names
genome = out$genome
chromosomes = out$chromosomes

# Save chromosome size
write.table(
  data.frame(a = chrom_names, b = as.integer(chrLen)),
  paste(data_dir, "simulated-genome-chrom-sizes.tsv", sep=""),
  sep="\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)


#randomly generate quality of reads
randomQuality <- function(read, ...) {
  paste(
    sample(
      unlist(strsplit(rawToChar(as.raw(33:126)),"")),
      length(read),
      replace = TRUE
    ), collapse=""
  )
}

defaultErrorProb1 <- function () {
  prob <- list(A = c(1,0,0,0), C = c(0,1,0,0), G = c(0,0,1,0), T = c(0,0,0,1))
  prob <- lapply(prob, "names<-", c("A", "C", "G", "T"))
  prob
}

readError <- function (
  read,
  qual,
  alphabet = c("A", "C", "G", "T"),
  prob = defaultErrorProb1(),
  ...
) {
  read <- gsub(paste("[^", paste(alphabet, collapse = "", sep = ""),
                     "]", sep = ""), alphabet[1], read)
  errorPos <- rep(1, length(qual)) < qual
  if (any(errorPos)) {
    for (i in which(errorPos)) {
      transProb <- prob[[substr(read, i, i)]]
      substr(read, i, i) <- sample(names(transProb), 1, prob = transProb)
    }
  }
  read
}

pos2fastq <- function (
  readPos,
  names,
  quality,
  sequence,
  qualityFun,
  errorFun,
  readLen = 36,
  file,
  qualityType = c("Illumina", "Sanger", "Solexa"),
  ...
) {
  if (file != "" && !is(file, "connection") && !file.exists(file))
    file <- file(file, open = "w", blocking = FALSE)
  replaceProb <- if (is.null(list(...)$prob))
    defaultErrorProb1()
  else match.fun(list(...)$prob)
  qualityFun <- match.fun(qualityFun)
  errorFun <- match.fun(errorFun)
  for (i in 1:2) {  # for each strand
    for (j in 1:length(readPos[[i]])) {  # for each chromosome
      readSeq <- ChIPsim::readSequence(
        readPos[[i]][j],
        sequence,
        strand = ifelse(i == 1, 1, -1),
        readLen = readLen
      )
      readQual <- qualityFun(readSeq, quality, ...)
      #readSeq <- errorFun(readSeq, decodeQuality(readQual, type = qualityType), prob = replaceProb)
      ChIPsim::writeFASTQ(
        as.character(readSeq),
        as.character(readQual),
        names[[i]][j],
        file = file,
        append = TRUE
      )
    }
  }
  invisible(sum(sapply(readPos, length)))
}


#generate read names
ReadNames <- paste("read_", 1:Nreads, sep="")

###################################################
### code chunk number 11: transitions
###################################################
transition <- list(
  BindingA=c(Background=1),
  BindingB=c(Background=1),
  BindingAB=c(Background=1),
  Background=c(
    BindingA=BindingAProb,
    BindingB=BindingBProb,
    BindingAB=BindingABProb,
    Background=BackgroundProb
  )
)
transition <- lapply(transition, "class<-", "StateDistribution")

###################################################
### code chunk number 12: initial
###################################################
init <- c(BindingA=0, BindingB=0, BindingAB=0, Background=1)
class(init) <- "StateDistribution"


###################################################
### code chunk number 13: bgEmission
###################################################
backgroundFeature <- function(start, length=BackgroundFeatureLength, shape=shapeBg, scale=scaleBg){
  weight <- rgamma(1, shape=shapeBg, scale=scaleBg)
  params <- list(start=start, length=length, weight=weight)
  class(params) <- c("Background", "SimulatedFeature")

  params
}


###################################################
### code chunk number 14: bindingEmission
###################################################
bindingFeature <- function (target, defaultShape, defaultScale, defaultEnrichment) {
  function(start, length=BindingFeatureLength, shape=defaultShape, scale=defaultScale, enrichment=defaultEnrichment, r=r1){
    stopifnot(r > 1)

    avgWeight <- shape * scale * enrichment
    lowerBound <- ((r - 1) * avgWeight)
    weight <- actuar::rpareto1(1, r, lowerBound)

    params <- list(start=start, length=length, weight=weight)
    class(params) <- c(paste("Binding", target, sep=""), "SimulatedFeature")

    params
  }
}


###################################################
### code chunk number 15: features1_1
###################################################
generator <- list(
  BindingA=bindingFeature("A", shapeA, scaleA, enrichmentA),
  BindingB=bindingFeature("B", shapeB, scaleB, enrichmentB),
  BindingAB=bindingFeature("AB", shapeAB, scaleAB, enrichmentAB),
  Background=backgroundFeature
)




###################################################
### code chunk number 20: featureDensity1
###################################################
constRegion <- function(weight, length) rep(weight, length)
featureDensity.BindingA <- function(feature, ...) constRegion(feature$weight, feature$length)
featureDensity.BindingB <- function(feature, ...) constRegion(feature$weight, feature$length)
featureDensity.BindingAB <- function(feature, ...) constRegion(feature$weight, feature$length)
featureDensity.Background <- function(feature, ...) constRegion(feature$weight, feature$length)




##################################################
## code chunk number 24: reconcileFeatures
##################################################
reconcileFeatures.TFExperiment <- function(features, ...){
  bindAIdx <- sapply(features, inherits, "BindingA")
  bindBIdx <- sapply(features, inherits, "BindingB")
  bindABIdx <- sapply(features, inherits, "BindingAB")

  if (any(bindAIdx))
    bindLength <- features[[min(which(bindAIdx))]]$length
  else if (any(bindBIdx))
    bindLength <- features[[min(which(bindBIdx))]]$length
  else if (any(bindABIdx))
    bindLength <- features[[min(which(bindABIdx))]]$length
  else
    bindLength <- 1

  lapply(features, function(f) {
    if(inherits(f, "Background"))
      f$weight <- f$weight / bindLength
    ## The next three lines (or something to this effect)
    ## are required by all 'reconcileFeatures' implementations.
    f$overlap <- 0
    currentClass <- class(f)
    class(f) <- c(
      currentClass[-length(currentClass)],
      "ReconciledFeature",
      currentClass[length(currentClass)]
    )
    f
  })
}



###################################################
### code chunk number 27: featureDensity2
###################################################
if (spiky) {
  spikify <- function(feature, ...){
    featDens <- numeric(feature$length)
    featDens[floor(feature$length/2)] <- feature$weight
    featDens
  }
  featureDensity.BindingA <- spikify
  featureDensity.BindingB <- spikify
  featureDensity.BindingAB <- spikify
}




###################################################
### code chunk number 29: fragmentLength
###################################################
fragLength <- function(x, minLength, maxLength, meanLength, ...){
  sd <- (maxLength - minLength) / 4
  prob <- dnorm(minLength:maxLength, mean = meanLength, sd = sd)
  prob <- prob/sum(prob)
  prob[x - minLength + 1]
}




###################################################
### code chunk number 32: readLoc2
###################################################
set.seed(seed2)

GenerateChipSeqFastqFiles <- function(ExpNo) {
  if(!missing(ExpNo)) {
    ExpNo = paste("-", ExpNo)
  } else {
    ExpNo = ""
  }
  message("Place features...")
  features <- ChIPsim::placeFeatures(
    generator,
    transition,
    init,
    start = 0,
    length = chrLen,
    globals=list(shape=shapeBg, scale=scaleBg),
    experimentType="TFExperiment",
    lastFeat=c(
      BindingA=FALSE,
      BindingB=FALSE,
      BindingAB=FALSE,
      Background=TRUE
    ),
    control=list(
      BindingA=list(length=BindLength),
      BindingB=list(length=BindLength),
      BindingAB=list(length=BindLength)
    )
  )
  bindingVec <- vector(mode="character")
  startVec <- vector(mode="numeric")
  endVec <- vector(mode="numeric")
  weightVec <- vector(mode="numeric")

  for(i in 1:length(features)) {
    bindingVec[i] <- class(features[[i]])[1]
    # Stupid one-indexed R language...
    startVec[i] <- features[[i]]$start - 1
    endVec[i] <- features[[i]]$start + features[[i]]$length - 1
    weightVec[i] <- features[[i]]$weight
  }

  FeaturesResult <- data.frame(cbind(
    chrom=rep("chr1"), start=startVec, end=endVec, name=bindingVec, score=weightVec
  ))
  write.table(
    FeaturesResult,
    paste(data_dir, "simulated-features-2-targets", ExpNo, ".bed", sep=""),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
  )

  # Baseline: the baseline consists of only background features (`0`)
  features0 <- features
  for (target in c("A", "B", "AB")) {
    for (i in which(sapply(features, inherits, paste("Binding", target, sep="")))) {
      # Overwrite both target features with background class and overwrite weights
      class(features0[[i]]) <- class(features0[[i+1]])
      features0[[i]]$weight <- features0[[i+1]]$weight
    }
  }

  # 1. target: the features for the first target (`1`)
  features1 <- features
  for (i in which(sapply(features, inherits, "BindingB"))) {
    # Overwrite target B features with background class and overwrite weights
    class(features1[[i]]) <- class(features1[[i+1]])
    features1[[i]]$weight <- features1[[i+1]]$weight
  }
  targetAClass = class(features1[[which(sapply(features, inherits, "BindingA"))[1]]])
  for (i in which(sapply(features, inherits, "BindingAB"))) {
    # Switch AB features to A features
    class(features1[[i]]) <- targetAClass
  }

  # 2. target: features for the second target (`2`)
  features2 <- features
  for (i in which(sapply(features, inherits, "BindingA"))) {
    # Overwrite target A features with background class and overwrite weights
    class(features2[[i]]) <- class(features2[[i+1]])
    features2[[i]]$weight <- features2[[i+1]]$weight
  }
  targetBClass = class(features1[[which(sapply(features, inherits, "BindingB"))[1]]])
  for (i in which(sapply(features, inherits, "BindingAB"))) {
    # Switch AB features to B features
    class(features2[[i]]) <- targetBClass
  }

  message("Densify features...")
  dens0 <- ChIPsim::feat2dens(features0, length=chrLen)
  dens1 <- ChIPsim::feat2dens(features1, length=chrLen)
  dens2 <- ChIPsim::feat2dens(features2, length=chrLen)

  readDens0 <- ChIPsim::bindDens2readDens(
    dens0,
    fragLength,
    bind=BindLength,
    minLength=MinLength,
    maxLength=MaxLength,
    meanLength=MeanLength
  )
  readDens1 <- ChIPsim::bindDens2readDens(
    dens1,
    fragLength,
    bind=BindLength,
    minLength=MinLength,
    maxLength=MaxLength,
    meanLength=MeanLength
  )
  readDens2 <- ChIPsim::bindDens2readDens(
    dens2,
    fragLength,
    bind=BindLength,
    minLength=MinLength,
    maxLength=MaxLength,
    meanLength=MeanLength
  )

  message("Sample reads...")
  readLoc0 <- ChIPsim::sampleReads(readDens0, Nreads)
  readLoc1 <- ChIPsim::sampleReads(readDens1, Nreads)
  readLoc2 <- ChIPsim::sampleReads(readDens2, Nreads)


  names0 <- list(
    paste("read", 1:length(readLoc0[[1]]), sep="_"),
    paste("read", (1+length(readLoc0[[1]])):(Nreads), sep="_")
  )
  names1 <- list(
    paste("read", 1:length(readLoc1[[1]]), sep="_"),
    paste("read", (1+length(readLoc1[[1]])):(Nreads), sep="_")
  )
  names2 <- list(
    paste("read", 1:length(readLoc2[[1]]), sep="_"),
    paste("read", (1+length(readLoc2[[1]])):(Nreads), sep="_")
  )

  message("Write background reads to fastq...")
  pos2fastq(
    readPos = readLoc0,
    names = names0,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError,
    readLen = LengthReads,
    file = paste(data_dir, "simulated-reads-input", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
  message("Write target 1 reads to fastq...")
  pos2fastq(
    readPos = readLoc1,
    names = names1,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError,
    readLen = LengthReads,
    file = paste(data_dir, "simulated-reads-chip-target-1", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
  message("Write target 2 reads to fastq...")
  pos2fastq(
    readPos = readLoc2,
    names = names2,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError,
    readLen = LengthReads,
    file = paste(data_dir, "simulated-reads-chip-target-2", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
}

message("Start simulation...")
start = proc.time()[3]
GenerateChipSeqFastqFiles()
duration = round((proc.time()[3] - start) / 60, 2)
message(paste("It took", duration, "minutes to simulate the reads."))
