import os
import hydra
import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from text_detect.model import LLMDetector

from loguru import logger
from data import DatasetManager


class LLMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(cfg):
    """Load and preprocess the data with train/val/test split."""
    logger.info(f"Loading data from {cfg.data.path}")
    logger.debug(f"CWD: {os.getcwd()}")

    data = pd.read_csv(cfg.data.latest_data_path)
    texts = data["text"].values
    labels = data["label"].values

    logger.info(f"Successfully loaded {len(texts)} samples")
    logger.debug(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,  # 20% for test
        random_state=cfg.seed,
        stratify=labels,  # Maintain label distribution
    )

    # Second split: separate train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=0.2,  # 20% of remaining data for validation
        random_state=cfg.seed,
        stratify=train_val_labels,  # Maintain label distribution
    )

    logger.info(f"Split data: {len(train_texts)} training, {len(val_texts)} validation, {len(test_texts)} test samples")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def train(cfg):
    # Get the path to the hydra output directory
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger.add(
        os.path.join(hydra_path, "train.log"),
        rotation="100 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    # logger options:
    # logger.debug("Used for debugging your code.")
    # logger.info("Informative messages from your code.")
    # logger.warning("Everything works but there is something to be aware of.")
    # logger.error("There's been a mistake with the process.")
    # logger.critical("There is something terribly wrong and process may terminate.")

    logger.info("Starting training pipeline")
    logger.debug(f"Configuration: {cfg}")

    # Set random seeds
    pl.seed_everything(cfg.seed)
    logger.info(f"Set random seed to {cfg.seed}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.model_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set this environment variable before importing tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Load and split data
    logger.info("Loading data")

    dataset_manager = DatasetManager(cfg.data.raw_data_path, cfg.data.processed_data_path)
    dataset_manager.set_version("latest")
    train_loader, val_loader, test_loader = dataset_manager.create_dataloader_from_set(tokenizer, cfg)

    # Initialize model
    logger.info("Initializing model")
    model = LLMDetector(cfg)

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    if cfg.training.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg.training.output_dir, "checkpoints"),
            filename="llm-detector-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.training.save_top_k,
        )
        callbacks.append(checkpoint_callback)
        logger.info("Added checkpoint callback")

    # Early stopping callback
    if cfg.training.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.patience,
            mode="min",
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Added early stopping callback with patience {cfg.training.patience}")

    logger.info(f"Callbacks: {callbacks}")


    # Initialize Weights & Biases logger
    logger_wandb = False
    if cfg.wandb.use_wandb:
        logger_wandb = WandbLogger(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            save_dir=cfg.training.output_dir,
            job_type="train",
            tags=["training"],
            config={cfg},
        )
        logger.info("Initialized W&B logger")
    else:
        logger.warning("W&B logging disabled")


    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1,
        logger=logger_wandb,
        log_every_n_steps=10,
        enable_checkpointing=False, # Disable default checkpointing of last trained epoch
        callbacks=callbacks,
        # gradient_clip_val=1.0, # Prevent exploding gradients
        enable_progress_bar=True,
        precision=cfg.training.precision,
    )

    logger.info(f"Using device: {trainer.strategy.root_device}")


    # Train model
    logger.info("Starting model training")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    logger.success("Training completed successfully")

    model_path = os.path.join(cfg.training.output_dir, cfg.training.model_name)

    # Save final model if needed
    if cfg.training.save_model:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model state dict to {model_path}")
    else:
        logger.warning("Model not saved")

    # Save model as W&B artifact
    if cfg.wandb.save_artifact:
        artifact = wandb.Artifact(
            name="...",
            type="model",
            description="...",
            metadata={"..."},
        )
        artifact.add_file(os.path.join(cfg.training.output_dir, cfg.training.model_name))
        wandb.log_artifact(artifact)
    else:
        logger.warning("Model artifact not saved to W&B artifact registry")


    # Test the model
    logger.info("Starting model testing")
    test_results = trainer.test(model=model, dataloaders=test_loader)

    # Log test results
    logger.info(f"Test Results: {test_results}")
    if cfg.wandb.use_wandb and logger_wandb:
        logger_wandb.log_metrics({"test_loss": test_results[0]["test_loss"], "test_acc": test_results[0]["test_acc"]})

    logger.success("Training and testing completed successfully")


if __name__ == "__main__":
    train()
