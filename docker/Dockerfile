FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ=Europe/Moscow
ENV TERM xterm-256color

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y \
	python3-pip \
	python3-tk \
	libboost-all-dev

RUN apt-get -y install git
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
RUN tar -xvf LJSpeech-1.1.tar.bz2
COPY mv.sh .
ENTRYPOINT mv.sh