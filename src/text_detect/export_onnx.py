import torch
import onnx
from transformers import AutoTokenizer
import wandb
from omegaconf import OmegaConf
import pytorch_lightning as pl
import os
from text_detect.wandb_functions import load_download_artifact_model

production_model_artifact_path = (
    "mlops_project_group_37/wandb-registry-Text detect registry/Trained LLM detector:production"
)

api_key = api_key = os.getenv("WANDB_API_KEY")
api = wandb.Api(api_key=api_key)
artifact = api.artifact(production_model_artifact_path)
artifact.download(root="models/downloads")

# Get wandb config
wb_config = artifact.logged_by().config
cfg = OmegaConf.create(wb_config)

file_name = artifact.files()[0].name
model_path = os.path.join("models/downloads", file_name)

_, model = load_download_artifact_model(cfg, api, production_model_artifact_path)


# Set random seeds
pl.seed_everything(cfg.seed)

# Load tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)
transformer_model_name = cfg.model.transformer_name

# Set the model to evaluation mode
model.eval()

# Dummy input for export
dummy_input = tokenizer("This is a sample input.", return_tensors="pt")

# Export the model to ONNX
model.to_onnx(
    file_path=model_path,
    input_sample=dummy_input,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

onnx_model = torch.onnx._export(model)

# Add metadata for the transformer model name
onnx_model.metadata_props.append(onnx.ModelMetadataProto(key="transformer_model_name", value=transformer_model_name))

# Save the ONNX model with metadata
onnx.save_model(onnx_model, "model_with_transformer_name.onnx")
