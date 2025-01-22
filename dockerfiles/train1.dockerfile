# Base image
FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs configs/
COPY data data/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

COPY pyproject.toml pyproject.toml

# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
# RUN pip install . --no-deps --no-cache-dir --verbose

# ENTRYPOINT ["python", "-u", "src/text_detect/train.py"]