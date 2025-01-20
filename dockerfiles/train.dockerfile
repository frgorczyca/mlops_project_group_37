# Import a base image so we don't have to start from scratch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel AS base

# Run a bunch of linux commands
RUN apt update && \
    apt install --no-install-recommends -y build essential gcc & \
    apt clean & rm -rf /var/lib/apt/lists/*

# Copy the essential files from our folder to docker container.
COPY requirements.txt requirements.txt
COPY src/ src/
COPY data/ data/
RUN ls -la

RUN pip install -r requirements.txt --no-cache-dir
RUN python --version

# Set entry point, i.e. which file we run with which argument when running the docker container.
# The -u flag makes it print to console rather than the docker log file.
ENTRYPOINT ["python", "-u", "src/text_detect/train.py"]
