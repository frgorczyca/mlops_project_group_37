# Base image
#FROM  nvcr.io/nvidia/pytorch:22.07-py3
FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/text_detect/train.py"]



# Base image
#FROM python:3.11-slim AS base

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*

#COPY src src/
#COPY requirements.txt requirements.txt
#COPY requirements_dev.txt requirements_dev.txt
#COPY README.md README.md
#COPY pyproject.toml pyproject.toml

#RUN pip install -r requirements.txt --no-cache-dir --verbose
#RUN pip install . --no-deps --no-cache-dir --verbose

#ENTRYPOINT ["python", "-u", "src/text_detect/train.py"]
