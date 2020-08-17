FROM continuumio/miniconda

RUN apt-get update && apt-get -y install \
	make \
	build-essential \
	zlib1g-dev \
	libbz2-dev \
	liblzma-dev \
	vim

WORKDIR /peax

COPY environment.yml /peax
RUN conda env create -f ./environment.yml

COPY ui ui
WORKDIR /peax/ui

SHELL ["conda", "run", "-n", "px", "/bin/bash", "-c"]

RUN npm install
RUN npm build

COPY start.py .
COPY server server

# ENTRYPOINT bash
# ENTRYPOINT python -d start.py -c 
