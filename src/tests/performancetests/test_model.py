import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import hydra
import os
import wandb
import time

from text_detect.data import LLMDataset, load_data
from text_detect.wandb_functions import (
    load_wandb_env_vars,
    load_download_artifact_model,
    cleanup_downloaded_model,
)

from dotenv import load_dotenv

load_dotenv()

print(os.getcwd())


def test_model():
    config_name = "default.yaml"

    artifact_registry_path = os.getenv("MODEL_NAME")
    print(f"Starting model test of {artifact_registry_path}")

    print(f"Loading config: {config_name}")
    hydra.initialize(config_path="../../../configs", version_base="1.1")
    cfg = hydra.compose(config_name=config_name)
    print(f"Config: {cfg}")

    # Set random seeds
    pl.seed_everything(cfg.seed)
    print(f"Set random seed to {cfg.seed}")

    # Load environment variables
    print("Loading W&B environment variables")
    api_key, team_name, project_name, _, _, _ = load_wandb_env_vars()

    # Initialize W&B API
    print("Initializing W&B API")
    api = wandb.Api(api_key=api_key)

    # Extract artifact name and version
    print("Extracting artifact name and version")
    artifact, model = load_download_artifact_model(cfg, api, artifact_registry_path)

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.model.transformer_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set this environment variable before importing tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)

    print(f"Loading test data: {cfg.data.test_path}")
    test_texts, test_labels = load_data(cfg.data.test_path)
    test_dataset = LLMDataset(texts=test_texts, labels=test_labels, tokenizer=tokenizer, max_length=cfg.data.max_length)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # Initialize trainer
    print("Initializing PyTorch Lightning trainer")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    print("Starting evaluation")
    start = time.time()
    test_results = trainer.test(model=model, dataloaders=test_loader)
    end = time.time()
    print(f"Time taken: {end - start}")
    print(f"Test results: {test_results[0]}")

    assert end - start < 60, "Evaluation took too long"
    assert test_results[0]["test_accuracy"] > 0.8, "Model accuracy is too low"

    print("Test passed")

    print("Cleaning up downloaded model")
    cleanup_downloaded_model()
    print("Downloaded model cleaned up")


if __name__ == "__main__":
    test_model()
