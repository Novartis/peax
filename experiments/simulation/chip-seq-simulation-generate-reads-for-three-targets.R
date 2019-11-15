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
enrichmentA <- 4.5
shapeB <- 1
scaleB <- 18
enrichmentB <- 5.1
shapeC <- 1.5
scaleC <- 15
enrichmentC <- 3.7
shapeAB <- 1
scaleAB <- 24
enrichmentAB <- 5
shapeAC <- 1.2
scaleAC <- 20
enrichmentAC <- 4.6
shapeBC <- 1.2
scaleBC <- 18
enrichmentBC <- 4.2
shapeABC <- 1.2
scaleABC <- 20
enrichmentABC <- 4.8
shapeBg <- 1
scaleBg <- 20
r1 <- 2
BindLength <- 50
MinLength <- 150
MaxLength <- 250
MeanLength <- 200
BindingAProb <- 0.006
BindingBProb <- 0.006
BindingCProb <- 0.006
BindingABProb <- 0.006
BindingACProb <- 0.006
BindingBCProb <- 0.006
BindingABCProb <- 0.006
BindingProb <- (
  BindingAProb + BindingBProb + BindingCProb +
  BindingABProb + BindingACProb + BindingBCProb +
  BindingABCProb
)
BackgroundProb <- 1-BindingProb
BackgroundFeatureLength <- 1000
BindingFeatureLength <- 500
seedGenome <- 1111
seedFeatures <- 1112
seedA <- 1234
seedB <- 1235
seedC <- 1236
# number of reads
Nreads <- 1e5
# length of reads
LengthReads <- 50
# generate random genome
set.seed(seedGenome)

args = commandArgs(trailingOnly = TRUE)

# length of the chromosome (which is the same as the length of the genome 120 Mb)
chrLen <- c(1.2e7)
if (length(args) >= 1) chrLen = c(as.integer(args[[1]]))
if (length(args) >= 2) Nreads = c(as.integer(args[[2]]))
if (length(args) >= 3) {
  peakDistortion = args[[3]]
  if (peakDistortion == "spiky") {
    postFix = "-spiky-peaks"
  } else if (peakDistortion == "distorted") {
    postFix = "-distorted-peaks"
  } else {
    peakDistortion = FALSE
    postFix = ""
  }
} else {
  peakDistortion = FALSE
  postFix = ""
}

suffix = paste("-", chrLen, "-", Nreads, sep="")

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
    file=paste(data_dir, "simulated-genome", suffix,".fa", sep=""),
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
      genome <- readDNAStringSet(
        paste(data_dir, "simulated-genome", suffix,".fa", sep="")
      )
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
  paste(data_dir, "simulated-genome-chrom-sizes", suffix, ".tsv", sep=""),
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
  BindingC=c(Background=1),
  BindingAB=c(Background=1),
  BindingAC=c(Background=1),
  BindingBC=c(Background=1),
  BindingABC=c(Background=1),
  Background=c(
    BindingA=BindingAProb,
    BindingB=BindingBProb,
    BindingC=BindingCProb,
    BindingAB=BindingABProb,
    BindingAC=BindingACProb,
    BindingBC=BindingBCProb,
    BindingABC=BindingABCProb,
    Background=BackgroundProb
  )
)
transition <- lapply(transition, "class<-", "StateDistribution")

###################################################
### code chunk number 12: initial
###################################################
init <- c(BindingA=0, BindingB=0, BindingC=0, BindingAB=0, BindingAC=0, BindingBC=0, BindingABC=0, Background=1)
class(init) <- "StateDistribution"


###################################################
### code chunk number 13: bgEmission
###################################################
backgroundFeature <- function(start, length=BackgroundFeatureLength, shape=shapeBg, scale=scaleBg){
  # Allow switching just the seed for the background weights
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
  BindingC=bindingFeature("C", shapeC, scaleC, enrichmentC),
  BindingAB=bindingFeature("AB", shapeAB, scaleAB, enrichmentAB),
  BindingAC=bindingFeature("AC", shapeAC, scaleAC, enrichmentAC),
  BindingBC=bindingFeature("BC", shapeBC, scaleBC, enrichmentBC),
  BindingABC=bindingFeature("ABC", shapeABC, scaleABC, enrichmentABC),
  Background=backgroundFeature
)




###################################################
### code chunk number 20: featureDensity1
###################################################
constRegion <- function(weight, length) rep(weight, length)
defaultFeatureDensity <- function(feature, ...) constRegion(feature$weight, feature$length)
featureDensity.BindingA <- defaultFeatureDensity
featureDensity.BindingB <- defaultFeatureDensity
featureDensity.BindingC <- defaultFeatureDensity
featureDensity.BindingAB <- defaultFeatureDensity
featureDensity.BindingAC <- defaultFeatureDensity
featureDensity.BindingBC <- defaultFeatureDensity
featureDensity.BindingABC <- defaultFeatureDensity
featureDensity.Background <- defaultFeatureDensity




##################################################
## code chunk number 24: reconcileFeatures
##################################################
reconcileFeatures.TFExperiment <- function(features, ...){
  bindAIdx <- sapply(features, inherits, "BindingA")
  bindBIdx <- sapply(features, inherits, "BindingB")
  bindCIdx <- sapply(features, inherits, "BindingC")
  bindABIdx <- sapply(features, inherits, "BindingAB")
  bindACIdx <- sapply(features, inherits, "BindingAC")
  bindBCIdx <- sapply(features, inherits, "BindingBC")
  bindABCIdx <- sapply(features, inherits, "BindingABC")

  if (
    any(bindAIdx) ||
    any(bindBIdx) ||
    any(bindCIdx) ||
    any(bindABIdx) ||
    any(bindACIdx) ||
    any(bindBCIdx) ||
    any(bindABCIdx)
  )
    bindLength <- BindLength
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
if (peakDistortion == "spiky") {
  spikify <- function(feature, ...){
    # Set density to 0
    featDens <- numeric(feature$length)
    # Add weight only to the very middle
    featDens[floor(feature$length/2)] <- feature$weight
    featDens
  }
  featureDensity.BindingA <- spikify
  featureDensity.BindingB <- spikify
  featureDensity.BindingC <- spikify
  featureDensity.BindingAB <- spikify
  featureDensity.BindingAC <- spikify
  featureDensity.BindingBC <- spikify
  featureDensity.BindingABC <- spikify
} else if (peakDistortion == "distorted") {
  # Inspired by
  # https://journals.plos.org/plosone/artice?id=10.1371/journal.pone.0060002
  featureDensity.BindingA <- function(feature, ...) {
    # Set density to 0
    featDens <- numeric(feature$length)
    # Right-shifted weights
    featDens[(feature$length - 7):feature$length] <- feature$weight
    featDens
  }
  featureDensity.BindingB <- function(feature, ...) {
    # Set density to 0
    featDens <- numeric(feature$length)
    # Left-shifted weights
    featDens[1:7] <- feature$weight
    featDens
  }
  featureDensity.BindingC <- function(feature, ...) {
    # Set density to 0
    featDens <- numeric(feature$length)
    # Add weight only to the very middle
    middle = floor(feature$length/2)
    featDens[(middle - 3):(middle + 3)] <- feature$weight
    featDens
  }
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
### Place features
###################################################
placeFeatures <- function(seedFeatures, seedBg) {
  set.seed(seedFeatures)
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
      BindingC=FALSE,
      BindingAB=FALSE,
      BindingAC=FALSE,
      BindingBC=FALSE,
      BindingABC=FALSE,
      Background=TRUE
    ),
    control=list(
      BindingA=list(length=BindLength),
      BindingB=list(length=BindLength),
      BindingC=list(length=BindLength),
      BindingAB=list(length=BindLength),
      BindingAC=list(length=BindLength),
      BindingBC=list(length=BindLength),
      BindingABC=list(length=BindLength)
    )
  )
  features
}




###################################################
### Adjust features by target
###################################################
adjustFeatures <- function(originalFeatures, targets, otherTargets, seedBgWeights) {
  set.seed(seedBgWeights)

  targetName = paste("Binding", targets[1], sep="")

  # Alter background weights
  targetFeatures <- originalFeatures
  for (i in which(sapply(targetFeatures, inherits, "Background"))) {
    targetFeatures[[i]]$weight <- rgamma(1, shape=shapeBg, scale=scaleBg) / BindLength
  }

  # Switch combo-binding to primary target
  targetClass = class(targetFeatures[[which(sapply(targetFeatures, inherits, targetName))[1]]])
  for (target in targets[2:length(targets)]) {
    for (i in which(sapply(originalFeatures, inherits, paste("Binding", target, sep="")))) {
      # Switch to the target feature class
      class(targetFeatures[[i]]) <- targetClass
    }
  }

  # Overwrite other targets with background
  for (target in otherTargets) {
    for (i in which(sapply(targetFeatures, inherits, paste("Binding", target, sep="")))) {
      # Overwrite both target features with background class and overwrite weights
      class(targetFeatures[[i]]) <- class(targetFeatures[[i+1]])
      targetFeatures[[i]]$weight <- targetFeatures[[i+1]]$weight
    }
  }

  # Create background-only features
  backgroundFeatures <- targetFeatures
  for (target in targets) {
    for (i in which(sapply(backgroundFeatures, inherits, paste("Binding", target, sep="")))) {
      # Overwrite both target features with background class and overwrite weights
      class(backgroundFeatures[[i]]) <- class(backgroundFeatures[[i+1]])
      backgroundFeatures[[i]]$weight <- backgroundFeatures[[i+1]]$weight
    }
  }

  list(target=targetFeatures, background=backgroundFeatures)
}




###################################################
### code chunk number 32: readLoc2
###################################################

GenerateChipSeqFastqFiles <- function(ExpNo) {
  if(missing(ExpNo)) {
    ExpNo = ""
  }
  message("Place features...")
  set.seed(seedFeatures)
  features <- placeFeatures(seedFeatures, seedFeatures)

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
    paste(data_dir, "simulated-features-3-targets", ExpNo, ".bed", sep=""),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
  )

  message("Adjust features...")
  featuresA <- adjustFeatures(
    features, c("A", "AB", "AC", "ABC"), c("B", "C", "BC"), seedA
  )
  featuresB <- adjustFeatures(
    features, c("B", "AB", "BC", "ABC"), c("A", "C", "AC"), seedB
  )
  featuresC <- adjustFeatures(
    features, c("C", "BC", "AC", "ABC"), c("A", "B", "AB"), seedC
  )



  message("")
  message("Generate reads for target A:")
  set.seed(seedA)

  message("Densify features...")
  dens0 <- ChIPsim::feat2dens(featuresA$background, length=chrLen)
  dens1 <- ChIPsim::feat2dens(featuresA$target, length=chrLen)
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

  message("Sample reads...")
  readLoc0 <- ChIPsim::sampleReads(readDens0, Nreads, strandProb=c(0.5, 0.5))
  readLoc1 <- ChIPsim::sampleReads(readDens1, Nreads, strandProb=c(0.42, 0.58))
  names0 <- list(
    paste("read", 1:length(readLoc0[[1]]), sep="_"),
    paste("read", (1+length(readLoc0[[1]])):(Nreads), sep="_")
  )
  names1 <- list(
    paste("read", 1:length(readLoc1[[1]]), sep="_"),
    paste("read", (1+length(readLoc1[[1]])):(Nreads), sep="_")
  )

  message("Write background reads to fastq...")
  pos2fastq(
    readPos = readLoc0,
    names = names0,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError,
    readLen = LengthReads,
    file = paste(data_dir, "reads-3-targets-target-1-background", postFix, ExpNo, ".fastq", sep=""),
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
    file = paste(data_dir, "reads-3-targets-target-1", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )



  message("")
  message("Generate reads for target B:")
  set.seed(seedB)

  message("Densify features...")
  dens0 <- ChIPsim::feat2dens(featuresB$background, length=chrLen)
  dens2 <- ChIPsim::feat2dens(featuresB$target, length=chrLen)
  readDens0 <- ChIPsim::bindDens2readDens(
    dens0,
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
  readLoc0 <- ChIPsim::sampleReads(readDens0, Nreads, strandProb=c(0.5, 0.5))
  readLoc2 <- ChIPsim::sampleReads(readDens2, Nreads, strandProb=c(0.6, 0.4))
  names0 <- list(
    paste("read", 1:length(readLoc0[[1]]), sep="_"),
    paste("read", (1+length(readLoc0[[1]])):(Nreads), sep="_")
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
    file = paste(data_dir, "reads-3-targets-target-2-background", postFix, ExpNo, ".fastq", sep=""),
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
    file = paste(data_dir, "reads-3-targets-target-2", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )



  message("")
  message("Generate reads for target C:")
  set.seed(seedC)

  message("Densify features...")
  dens0 <- ChIPsim::feat2dens(featuresC$background, length=chrLen)
  dens3 <- ChIPsim::feat2dens(featuresC$target, length=chrLen)
  readDens0 <- ChIPsim::bindDens2readDens(
    dens0,
    fragLength,
    bind=BindLength,
    minLength=MinLength,
    maxLength=MaxLength,
    meanLength=MeanLength
  )
  readDens3 <- ChIPsim::bindDens2readDens(
    dens3,
    fragLength,
    bind=BindLength,
    minLength=MinLength,
    maxLength=MaxLength,
    meanLength=MeanLength
  )

  message("Sample reads...")
  readLoc0 <- ChIPsim::sampleReads(readDens0, Nreads, strandProb=c(0.5, 0.5))
  readLoc3 <- ChIPsim::sampleReads(readDens3, Nreads, strandProb=c(0.5, 0.5))
  names0 <- list(
    paste("read", 1:length(readLoc0[[1]]), sep="_"),
    paste("read", (1+length(readLoc0[[1]])):(Nreads), sep="_")
  )
  names3 <- list(
    paste("read", 1:length(readLoc3[[1]]), sep="_"),
    paste("read", (1+length(readLoc3[[1]])):(Nreads), sep="_")
  )

  message("Write background reads to fastq...")
  pos2fastq(
    readPos = readLoc0,
    names = names0,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError,
    readLen = LengthReads,
    file = paste(data_dir, "reads-3-targets-target-3-background", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
  message("Write target 3 reads to fastq...")
  pos2fastq(
    readPos = readLoc3,
    names = names3,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError,
    readLen = LengthReads,
    file = paste(data_dir, "reads-3-targets-target-3", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
}

message("Start simulation...")
start = proc.time()[3]
GenerateChipSeqFastqFiles(suffix)
duration = round((proc.time()[3] - start) / 60, 2)
message(paste("It took", duration, "minutes to simulate the reads."))
