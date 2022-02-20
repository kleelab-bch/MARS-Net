FROM tensorflow/tensorflow:2.4.3-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    nano

# Add new user to avoid running as root
RUN useradd -ms /bin/bash docker
USER docker
WORKDIR /home/docker/MARS-Net

# Copy this version of of the model garden into the image
COPY --chown=docker . /home/docker/MARS-Net

ENV PATH="/home/docker/.local/bin:${PATH}"

RUN pip install -r requirements.txt
