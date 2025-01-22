import os
import torch
from transformers import AutoTokenizer
from text_detect.wandb_functions import load_wandb_env_vars, load_download_artifact_model, cleanup_downloaded_model
from omegaconf import OmegaConf
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader, Dataset
import typer


def inference(
    input_text: str = typer.Argument(..., help="Text to run inference on"),
    model_alias: str = typer.Option(
        "production",
        "--model-alias",
        "-m",
        help="Model alias to use from wandb registry (e.g., 'production', 'latest', 'v1', etc.).",
    ),
):
    """Run inference on the alias model with the input text."""
    production_model_artifact_path = (
        f"mlops_project_group_37/wandb-registry-Text detect registry/Trained LLM detector:{model_alias}"
    )

    api_key, _, _, _, _, _ = load_wandb_env_vars()
    api = wandb.Api(api_key=api_key)
    artifact = api.artifact(production_model_artifact_path)

    # Get your wandb config
    wb_config = artifact.logged_by().config

    # Convert to OmegaConf/Hydra config
    cfg = OmegaConf.create(wb_config)

    _, model = load_download_artifact_model(cfg, api, production_model_artifact_path)

    # Set random seeds
    pl.seed_everything(cfg.seed)
    print(f"Set random seed to {cfg.seed}")

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.model.transformer_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set this environment variable before importing tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)

    encoded_input = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=cfg.data.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Create a DataLoader
    class SinglePredictionDataset(Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[0], "attention_mask": self.attention_mask[0]}

    dataset = SinglePredictionDataset(encoded_input["input_ids"], encoded_input["attention_mask"])
    predict_dataloader = DataLoader(dataset, batch_size=1)

    trainer = pl.Trainer(accelerator="auto", devices=1)

    # Run inference
    predictions = trainer.predict(model, predict_dataloader)

    logits = predictions[0].squeeze()  # Remove batch dimension

    # Handle batch dimension properly
    probabilities = torch.softmax(logits, dim=-1)

    class_names = ["Human", "AI"]

    predicted_class = logits.argmax(-1).item()
    probs_array = probabilities.detach().numpy()  # Convert to numpy array
    human_prob = probs_array[0]  # Probability for Human
    ai_prob = probs_array[1]  # Probability for AI

    print(f"Input text: {input_text}")
    # print(f"Enconded input: {encoded_input}")
    print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
    print(f"Probabilities - Human: {human_prob}, AI: {ai_prob}), Sum: {human_prob + ai_prob}")
    print(f"Raw logits: {logits}")

    cleanup_downloaded_model()


if __name__ == "__main__":
    typer.run(inference)
