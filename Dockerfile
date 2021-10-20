FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install python3.8 python3-pip -y

RUN mkdir /tmp/sources
COPY setup.py /tmp/sources/setup.py
COPY servier /tmp/sources/servier

RUN ls /tmp/sources
RUN pip install /tmp/sources