FROM pytorch/pytorch:latest

RUN apt-get -y update

COPY /src /src
COPY requirements_dev.txt /tmp/requirements_dev.txt

RUN python -m pip install -r /tmp/requirements_dev.txt --quiet --no-cache-dir \
    && rm -f /tmp/requirements_dev.txt

WORKDIR /src