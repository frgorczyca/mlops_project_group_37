#Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3

#Install python
RUN apt update && apt install --no-install-recommends -y build-essential gcc && apt clean && rm -rf /var/lib/apt/lists/*
# Before the lines were common for any Docker application where python is run.
#All the next are to some degree speciifc lines for the application we want to run


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/test_mlops/evaluate.py"]
