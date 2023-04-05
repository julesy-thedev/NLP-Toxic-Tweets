FROM python:3

WORKDIR /usr/src/app

RUN export DEBIAN_FRONTEND=noninteractive

RUN python -m pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt