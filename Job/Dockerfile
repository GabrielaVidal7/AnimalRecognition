# 1. Get the base image
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

LABEL Gabriela Barion Vidal (gabrielavidal)

# 2. Set variables
ENV PATH /usr/local/cuda/bin/:$PATH
#ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/darknet
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/site-packages/torch/lib/nvidia/cublas/lib/:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

LABEL com.nvidia.volumes.needed="nvidia_driver"


# 3. Install all the softwares needed
# Install and update apt-key because of a bug
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt -y update
RUN apt -y full-upgrade
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC-3 apt-get -y install tzdata
# Install git to clone the repository
RUN apt-get install -y git

# 4. Install dependencies and lybraries
# Upgrade python
RUN apt install -y python3 --upgrade
# Install pip
RUN apt install -y python3-pip
# Installing all lybraries needed in main program
RUN pip install torch torchvision
RUN pip install sentence-transformers
RUN pip install -U imbalanced-learn
RUN pip install ktrain