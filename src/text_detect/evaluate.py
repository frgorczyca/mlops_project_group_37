import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import hydra
import pandas as pd
import os

from text_detect.model import LLMDetector
from text_detect.train import LLMDataset  # Import your dataset class

@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def evaluate(cfg):
    # Load test data
    test_data = pd.read_csv("path/to/test_data.csv")
    texts = test_data['text'].values
    labels = test_data['label'].values

    # Initialize model and load weights
    model = LLMDetector(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.training.checkpoint_dir, cfg.training.model_name)))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Create test dataset and dataloader
    test_dataset = LLMDataset(texts, labels, tokenizer, max_length=cfg.data.max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize Weights & Biases logger
    logger_wandb = False

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=logger_wandb,
        enable_progress_bar=True,
    )

    # Test the model
    trainer.test(model, test_loader)

if __name__ == "__main__":
    evaluate()
