# inference stage
FROM ubuntu:20.04 As inference

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg git python3 python3-pip unzip && \
    pip3 install --upgrade pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt && rm requirements.txt

RUN mkdir /app 
COPY . /app

COPY asserts.zip /tmp/
RUN unzip /tmp/asserts.zip -d /app/ && rm -r /app/asserts/examples /tmp/asserts.zip

WORKDIR /app

# training stage
FROM ubuntu:20.04 As training

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg git python3 python3-pip unzip && \
    pip3 install --upgrade pip

COPY requirements_training.txt .

RUN pip3 install -r requirements_training.txt && rm requirements_training.txt

RUN mkdir /app 
COPY . /app

COPY asserts.zip /tmp/
RUN unzip /tmp/asserts.zip -d /app/ && rm -r /app/asserts/examples /tmp/asserts.zip

WORKDIR /app
