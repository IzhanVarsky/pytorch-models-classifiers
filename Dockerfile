FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git

RUN apt-get update -y
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip cmake
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y cmake

RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY ./requirements.txt ./requirements.txt
WORKDIR .
RUN pip3 install -r ./requirements.txt

ENTRYPOINT ["./entry.sh"]