FROM nvcr.io/nvidia/pytorch:23.11-py3
RUN apt-get update
RUN pip3 install matplotlib
RUN apt-get install cmake -y
RUN mkdir /cusync
COPY . /cusync/