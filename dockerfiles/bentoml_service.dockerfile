FROM python:3.11-slim
WORKDIR /bento
COPY /src .
COPY /configs .
RUN pip install onnxruntime numpy bentoml bentoml transformers torch pytorch-lightning wandb loguru
CMD ["bentoml", "serve", "bentoml_service:TextDetectionService"]
