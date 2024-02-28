# Specifying the base image with an explicit platform (e.g., linux/amd64)
FROM --platform=linux/arm64/v8 pytorch/pytorch:latest
LABEL authors="vardanyan1"

# Set non-interactive frontend (avoids some issues during building)
ENV DEBIAN_FRONTEND=noninteractive

# Install required linux libraries in one RUN to reduce layers
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    gcc \
    gfortran \
    build-essential \
    libblas-dev \
    liblapack-dev \
    cmake \
    mlocate \
    && rm -rf /var/lib/apt/lists/*

# Install Multinest:
ENV LD_LIBRARY_PATH=/workspace/MultiNest/MultiNest_v3.12_CMake/multinest/lib/:$LD_LIBRARY_PATH
RUN git clone https://github.com/farhanferoz/MultiNest && \
    cd MultiNest/MultiNest_v3.12_CMake/multinest && \
    cmake . && \
    make

# Install Python dependencies
RUN pip3 install h5py==3.9.0 numpy==1.24.3 scipy==1.10.1 \
    pymultinest==2.12 astropy==5.3.2 psycopg2-binary requests

ENV INFERENCE_PATH=/workspace/MultiNest/MultiNest_v3.12_CMake/multinest/files/inference_files
ENV MODEL_PATH=/workspace/MultiNest/MultiNest_v3.12_CMake/multinest/files/inference_files/final_model_V3

WORKDIR MultiNest/MultiNest_v3.12_CMake/multinest

# Copy the Python scripts to the container

COPY multinest_modeling.py .
COPY ./files ./files
COPY ./data.csv ./data.csv

# Set the entrypoint command to run the Python script

CMD ["python3", "multinest_modeling2.py"]
