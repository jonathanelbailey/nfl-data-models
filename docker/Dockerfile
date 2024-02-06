FROM python:3.12.0-slim-bullseye
LABEL authors="Jonathan Bailey"

WORKDIR /usr/src/app

RUN apt update -y && apt install build-essential libsnappy-dev -y  --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ['python']