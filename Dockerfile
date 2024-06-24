FROM ubuntu:22.04

WORKDIR /IA368

RUN apt update && apt upgrade -y && apt install python3 pip swig -y

RUN pip install torch opencv-python

COPY tella/ ./tella

RUN cd tella && pip install -e ".[dev]" && pip install numpy==1.21.4

COPY project/ ./project

COPY exec.sh .
