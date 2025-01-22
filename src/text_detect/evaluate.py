import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import hydra
import os
import typer
import wandb

from text_detect.data import LLMDataset, load_data
from text_detect.wandb_functions import (
    load_wandb_env_vars,
    get_artifact_project_path,
    load_download_artifact_model,
    cleanup_downloaded_model,
)

from loguru import logger

from dotenv import load_dotenv

load_dotenv()


logger.add(
    "logs/evaluate.log", rotation="100 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def evaluate(
    artifact: str = typer.Argument(..., help="Artifact path in format 'name:version'"),
    config_name: str = typer.Option("default", "--config", "-c", help="Name of config file to use"),
) -> None:
    """
    Evaluate a model using the specified artifact and config.

    Args:
        artifact: Artifact path in format 'name:version'
        config_name: Name of the config file to configure the model with (without .yaml extension)
    """
    logger.info(f"Starting evaluation of model {artifact}")

    logger.info(f"Loading config: {config_name}")
    hydra.initialize(config_path="../../configs", version_base="1.1")
    cfg = hydra.compose(config_name=config_name)
    logger.debug(f"Config: {cfg}")

    # Set random seeds
    pl.seed_everything(cfg.seed)
    logger.info(f"Set random seed to {cfg.seed}")

    # Load environment variables
    logger.info("Loading W&B environment variables")
    api_key, team_name, project_name, _, _, _ = load_wandb_env_vars()

    # Initialize W&B API
    logger.info("Initializing W&B API")
    api = wandb.Api(api_key=api_key)

    # Extract artifact name and version
    logger.info("Extracting artifact name and version")
    artifact_name, artifact_name_version = artifact.split(":")
    artifact_project_path = get_artifact_project_path(team_name, project_name, artifact_name, artifact_name_version)

    # Load and download the model
    logger.info(f"Loading and downloading model: {artifact_project_path}")
    artifact, model = load_download_artifact_model(cfg, api, artifact_project_path)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.transformer_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set this environment variable before importing tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)

    logger.info(f"Loading test data: {cfg.data.test_path}")
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
    logger.info("Initializing PyTorch Lightning trainer")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    logger.info("Starting evaluation")
    test_results = trainer.test(model=model, dataloaders=test_loader)
    logger.info(f"Test results: {test_results[0]}")

    # Save test results to W&B artifact metadata
    logger.info("Saving test results to W&B artifact metadata")
    for metric_name, metric_value in test_results[0].items():
        artifact.metadata[metric_name] = metric_value
    artifact.save()

    if True:  # Set to True to cleanup downloaded model
        logger.info("Cleaning up downloaded model")
        cleanup_downloaded_model()
        logger.info("Downloaded model cleaned up")

    logger.info("Evaluation complete")


if __name__ == "__main__":
    typer.run(evaluate)
