import os
import hydra
import torch
import pytorch_lightning as pl
import wandb
import sys

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from text_detect.model import LLMDetector
from text_detect.data import LLMDataset, load_data

from loguru import logger
from omegaconf import OmegaConf

from dotenv import load_dotenv
load_dotenv()


def setup_callbacks(cfg, logger):
    """Setup PyTorch Lightning callbacks based on configuration"""

    # Setup callbacks
    callbacks = []

    # Checkpoint callback (saving model)
    if cfg.training.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.training.output_dir,
            filename="llm-detector-{epoch:02d}-{val_accuracy:.2f}",
            monitor="val_accuracy",
            mode="max",
            save_top_k=cfg.training.save_top_k,
            save_weights_only=False,  # This ensures full checkpoint saving
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

    return callbacks


def initialize_wandb_logger(cfg, logger, model, train_texts, val_texts):
    """Initialize Weights & Biases logger if enabled"""

    if cfg.logging.enable_wandb:
        api_key = os.getenv("WANDB_API_KEY")
        team_name = os.getenv("WANDB_TEAM")
        project_name = os.getenv("WANDB_PROJECT")

        run_name = f"{cfg.optimizer.type}-lr{cfg.optimizer.lr}-bs{cfg.training.batch_size}"

        # Log into W&B
        wandb.login(key=api_key, relogin=True)

        # Initialize W&B logger
        wandb_logger = WandbLogger(
            entity=team_name,
            project=project_name,
            name=run_name,
            save_dir=cfg.training.output_dir,
            log_model=False,  # Disable automatic logging of models, since it doesn't work as intended
            job_type="train",
            tags=["training"],
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        # Log metadata
        wandb_logger.experiment.config.update({
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
        })

        # Watch model
        wandb_logger.watch(model, log="all") # log gradients, parameter histogram and model topology, remove "all" if only log gradients
        logger.info("Initialized W&B logger")
    else:
        wandb_logger = False
        logger.warning("W&B logging disabled")

    return wandb_logger


@hydra.main(version_base="1.1", config_path="../../configs", config_name="default")
def train(cfg):
    # Get the path to the hydra output directory
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger.add(
        os.path.join(hydra_path, "train.log"),
        rotation="100 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    logger.info("Starting training pipeline")
    logger.debug(f"Configuration: {cfg}")

    # Set random seeds
    pl.seed_everything(cfg.seed)
    logger.info(f"Set random seed to {cfg.seed}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.transformer_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set this environment variable before importing tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)
    
    # Load and split data
    logger.info("Loading data")
    train_val_texts, train_val_labels = load_data(cfg.data.train_path)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=cfg.training.val_size,
        random_state=cfg.seed,
        stratify=train_val_labels
    )

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = LLMDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer, max_length=cfg.data.max_length)
    val_dataset = LLMDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer, max_length=cfg.data.max_length)

    # Create dataloaders
    logger.info("Initializing dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )


    # Initialize model
    logger.info("Initializing model")
    model = LLMDetector(cfg)

    # Setup callbacks
    callbacks = setup_callbacks(cfg, logger)

    # Initialize Weights & Biases logger
    wandb_logger = initialize_wandb_logger(cfg, logger, model, train_texts, val_texts)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_checkpointing=cfg.training.save_checkpoints,
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

    # Summarize training results
    if cfg.training.save_checkpoints:
        checkpoint_callback = callbacks[0]
        logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
        # Retrieve the best validation accuracy
        best_val_accuracy = checkpoint_callback.best_model_score
        logger.info(f"Best Validation Accuracy: {best_val_accuracy}")

        # Log to W&B if using WandbLogger
        if cfg.logging.enable_wandb:
            wandb_logger.experiment.summary["best_val_accuracy"] = best_val_accuracy
            logger.info("Logged best validation accuracy to W&B")

            # Use Artifacts API to track model versions explicitly
            artifact = wandb.Artifact(
                name="llm-detector-model", 
                type="model",
                description="Model artifact for LLM Detector with best validation accuracy",
                metadata={"best_val_accuracy": float(best_val_accuracy)
            })
            
            artifact.add_file(checkpoint_callback.best_model_path)
            wandb_logger.experiment.log_artifact(artifact)
            logger.info("Logged best model to W&B Artifacts")


if __name__ == "__main__":
    train()
