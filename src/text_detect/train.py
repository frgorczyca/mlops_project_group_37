import os
import hydra
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from text_detect.model import LLMDetector

from loguru import logger


# Configure loguru logger
logger.add(
    "logs/training_{time}.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Set this environment variable before importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(cfg):
    """Load and preprocess the data with train/val/test split."""
    logger.info(f"Loading data from {cfg.data.path}")
    logger.debug(f"CWD: {os.getcwd()}")

    data = pd.read_csv(cfg.data.path)
    texts = data['text'].values
    labels = data['label'].values

    logger.info(f"Successfully loaded {len(texts)} samples")
    logger.debug(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=0.2,  # 20% for test
        random_state=cfg.seed,
        stratify=labels  # Maintain label distribution
    )

    # Second split: separate train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=0.2,  # 20% of remaining data for validation
        random_state=cfg.seed,
        stratify=train_val_labels  # Maintain label distribution
    )

    logger.info(f"Split data: {len(train_texts)} training, {len(val_texts)} validation, {len(test_texts)} test samples")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def train(cfg):
    logger.info("Starting training pipeline")
    logger.debug(f"Configuration: {cfg}")

    # Set random seeds
    pl.seed_everything(cfg.seed)
    logger.info(f"Set random seed to {cfg.seed}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.model_name}")
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Load and split data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_data(cfg)

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = LLMDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_length=cfg.data.max_length
    )
    val_dataset = LLMDataset(
        val_texts,
        val_labels,
        tokenizer,
        max_length=cfg.data.max_length
    )
    test_dataset = LLMDataset(
        test_texts,
        test_labels,
        tokenizer,
        max_length=cfg.data.max_length
    )

    # Create dataloaders
    logger.info("Initializing dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

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

    # Initialize logger
    logger_wandb = None
    if cfg.wandb.use_wandb:
        logger_wandb = WandbLogger(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            save_dir=cfg.training.output_dir,
            job_type="train",
            tags=["training"],
            config={
                cfg
            }
        )
        logger.info("Initialized W&B logger")

    # Get the path to the hydra output directory
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "train.log"))
    logger.info(cfg)

    # logger.debug("Used for debugging your code.")
    # logger.info("Informative messages from your code.")
    # logger.warning("Everything works but there is something to be aware of.")
    # logger.error("There's been a mistake with the process.")
    # logger.critical("There is something terribly wrong and process may terminate.")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="mps", # 'auto',
        devices=1,
        # devices=cfg.training.devices,
        logger=logger_wandb,
        log_every_n_steps=10,
        callbacks=callbacks,
        # enable_checkpointing=True,
        # detect_anomaly=True,
        # limit_train_batches=0.2
        gradient_clip_val=1.0, # Prevent exploding gradients
        enable_progress_bar=True,
        accumulate_grad_batches=2,  # Effective batch size = batch_size * accumulate
        precision="16-mixed", # Use mixed precision training
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

    # Test the model
    logger.info("Starting model testing")
    test_results = trainer.test(model=model, dataloaders=test_loader)

    # Log test results
    logger.info(f"Test Results: {test_results}")
    if cfg.wandb.use_wandb and logger_wandb:
        logger_wandb.log_metrics({
            "test_loss": test_results[0]["test_loss"],
            "test_acc": test_results[0]["test_acc"]
        })

    # Save final model if needed
    if cfg.training.save_model:
        save_path = f"models/{cfg.model.save_model_name}"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model state dict to {save_path}")

    logger.success("Training and testing completed successfully")

if __name__ == "__main__":
    train()
