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
shape1 <- 1
scale1 <- 20
enrichment1 <- 5
r1 <- 2
BindLength <- 50
MinLength <- 150
MaxLength <- 250
MeanLength <- 200
BindingProb <- 0.006
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
if (length(args) >= 2) {
  spiky = as.logical(args[[2]])
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

readError1 <- function (
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

pos2fastq1 <- function (
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
transition <- list(Binding=c(Background=1), Background=c(Binding=BindingProb, Background=BackgroundProb))
transition <- lapply(transition, "class<-", "StateDistribution")

transition0 <- list(Binding=c(Background=1), Background=c(Binding=0, Background=1))
transition0 <- lapply(transition0, "class<-", "StateDistribution")


###################################################
### code chunk number 12: initial
###################################################
init <- c(Binding=0, Background=1)
class(init) <- "StateDistribution"


###################################################
### code chunk number 13: bgEmission
###################################################
backgroundFeature <- function(start, length=BackgroundFeatureLength, shape=shape1, scale=scale1){
  weight <- rgamma(1, shape=shape1, scale=scale1)
  params <- list(start = start, length = length, weight = weight)
  class(params) <- c("Background", "SimulatedFeature")

  params
}


###################################################
### code chunk number 14: bindingEmission
###################################################
bindingFeature <- function(start, length=BindingFeatureLength, shape=shape1, scale=scale1, enrichment=enrichment1, r=r1){
  stopifnot(r > 1)

  avgWeight <- shape * scale * enrichment
  lowerBound <- ((r - 1) * avgWeight)
  weight <- actuar::rpareto1(1, r, lowerBound)

  params <- list(start = start, length = length, weight = weight)
  class(params) <- c("Binding", "SimulatedFeature")

  params
}


###################################################
### code chunk number 15: features1_1
###################################################

generator <- list(Binding=bindingFeature, Background=backgroundFeature)

###################################################
### code chunk number 20: featureDensity1
###################################################
constRegion <- function(weight, length) rep(weight, length)
featureDensity.Binding <- function(feature, ...) constRegion(feature$weight, feature$length)
featureDensity.Background <- function(feature, ...) constRegion(feature$weight, feature$length)




##################################################
## code chunk number 24: reconcileFeatures
##################################################
reconcileFeatures.TFExperiment <- function(features, ...){
  bindIdx <- sapply(features, inherits, "Binding")
  if(any(bindIdx))
    bindLength <- features[[min(which(bindIdx))]]$length
  else bindLength <- 1
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
  featureDensity.Binding <- function(feature, ...){
    featDens <- numeric(feature$length)
    featDens[floor(feature$length/2)] <- feature$weight
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
    globals=list(shape=shape1, scale=scale1),
    experimentType="TFExperiment",
    lastFeat=c(Binding = FALSE, Background = TRUE),
    control=list(Binding=list(length=BindLength))
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
    paste(data_dir, "simulated-features", ExpNo, ".bed", sep=""),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE,
  )

  # create features for the input data (i.e., the baseline)
  features0 <- features
  bindIdx <- sapply(features, inherits, "Binding")
  for(i in which(bindIdx)) {
    class(features0[[i]]) <- class(features0[[i+1]])
    features0[[i]]$weight <- features0[[i+1]]$weight
  }

  message("Densify features...")
  dens <- ChIPsim::feat2dens(features, length = chrLen)
  dens0 <- ChIPsim::feat2dens(features0, length = chrLen)

  readDens <- ChIPsim::bindDens2readDens(
    dens,
    fragLength,
    bind = BindLength,
    minLength = MinLength,
    maxLength = MaxLength,
    meanLength = MeanLength
  )
  readDens0 <- ChIPsim::bindDens2readDens(
    dens0,
    fragLength,
    bind = BindLength,
    minLength = MinLength,
    maxLength = MaxLength,
    meanLength = MeanLength
  )

  message("Sample reads...")
  readLoc <- ChIPsim::sampleReads(readDens, Nreads)
  readLoc0 <- ChIPsim::sampleReads(readDens0, Nreads)


  names <- list(
    paste("read", 1:length(readLoc[[1]]), sep="_"),
    paste("read", (1+length(readLoc[[1]])):(Nreads), sep="_")
  )
  names0 <- list(
    paste("read", 1:length(readLoc0[[1]]), sep="_"),
    paste("read", (1+length(readLoc0[[1]])):(Nreads), sep="_")
  )

  message("Write reads to fastq...")
  pos2fastq1(
    readPos = readLoc,
    names = names,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError1,
    readLen = LengthReads,
    file = paste(data_dir, "simulated-reads-chip", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
  pos2fastq1(
    readPos = readLoc0,
    names = names0,
    sequence = chromosomes,
    qualityFun = randomQuality,
    errorFun = readError1,
    readLen = LengthReads,
    file = paste(data_dir, "simulated-reads-input", postFix, ExpNo, ".fastq", sep=""),
    qualityType = c("Illumina")
  )
}

message("Start simulation...")
start = proc.time()[3]
GenerateChipSeqFastqFiles()
message(paste("It took", proc.time()[3] - start, "seconds to simulate the reads."))
