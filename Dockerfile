# base image
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \ 
      vim \             
&& rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN python3 -m pip install --upgrade pip

# install python libraries
RUN python3 -m pip install \
    numpy==2.2.6 \
    torch==2.9.0

# set working directory
WORKDIR /src

# copy source directory
COPY src/ ./src/