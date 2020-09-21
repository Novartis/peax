FROM continuumio/miniconda

RUN apt-get update && apt-get -y install \
	make \
	build-essential \
	zlib1g-dev \
	libbz2-dev \
	liblzma-dev \
	vim

WORKDIR /peax

RUN conda install python=3.7 \
				  cython \
				  cytoolz \
				  seaborn \
				  flask \
				  flask-cors \
				  nodejs=10.* \
				  scikit-learn=0.22.0 \
				  pandas \
				  pywget \
				  bokeh \
				  pydot \
				  h5py \
				  testpath==0.4.2 \
				  tqdm \
				  matplotlib \
				  requests \
				  statsmodels \
				  tensorflow \
				  pip
RUN conda install -c bioconda bwa \
							  samtools \
							  bedtools \
							  ucsc-bedtobigbed \
							  ucsc-fetchchromsizes \
							  deeptools \
							  pysam==0.15.3

RUN conda install -c conda-forge umap-learn \
								 tsfresh \
								 tslearn


RUN pip install apricot-select \
	cooler \
	higlass-python==0.2.1 \
	hnswlib==0.3.4 \
	ipywidgets==7.5.1 \
	joblib==0.14.0 \
	jupyterlab==1.1.1 \
	negspy \
	numba==0.46.0 \
	pybbi \
	pytest==5.3.1 \
	keras-tqdm \
	fastdtw \
	stringcase

RUN conda install llvmlite==0.32.1

COPY ui ui
WORKDIR /peax/ui

RUN npm install
RUN npm build

WORKDIR /peax

COPY start.py .
COPY server server

