import os
import hydra
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from text_detect.model import LLMDetector


## V V V V REPLACE DATASTUFF WITH YOUR OWN IMPLEMENTATION V V V V
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
    """Load and preprocess the data."""
    # This is a placeholder - implement your data loading logic here
    # For example:
    # texts = pd.read_csv(cfg.data.path)['text'].values
    # labels = pd.read_csv(cfg.data.path)['label'].values
    
    # Placeholder data
    texts = ["Sample text 1", "Sample text 2"]  # Replace with your data
    labels = [0, 1]  # Replace with your labels
    
    return train_test_split(
        texts, labels, 
        test_size=cfg.data.test_size,
        random_state=cfg.seed
    )
# ^ ^ ^ ^ REPLACE DATASTUFF WITH YOUR OWN IMPLEMENTATION ^ ^ ^ ^


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg):
    # Set random seeds
    pl.seed_everything(cfg.seed)
    
    # Load tokenizer
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    
    # Load and split data
    train_texts, val_texts, train_labels, val_labels = load_data(cfg)
    
    # Create datasets
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True, # Good practice to shuffle training data
        num_workers=cfg.training.num_workers,
        persistent_workers=True # Enable persistent workers for faster data loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=cfg.training.num_workers,
        persistent_workers=True # Enable persistent workers for faster data loading
    )
    
    # Initialize model
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
    
    # Early stopping callback
    if cfg.training.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.patience,
            mode="min",
        )
        callbacks.append(early_stopping_callback)
    
    # Initialize logger
    logger = None
    if cfg.logging.use_wandb:
        logger = WandbLogger(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            save_dir=cfg.training.output_dir,
            job_type="train",
            tags=["training"],
            config={
                cfg
            }
        )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='mps' if torch.backends.mps.is_available() else 'auto',
        devices=1,
        # devices=cfg.training.devices,
        logger=logger,
        log_every_n_steps=10,
        callbacks=callbacks,
        # enable_checkpointing=True,
        # detect_anomaly=True,
        # limit_train_batches=0.2
        enable_progress_bar=True,
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Add save model code here

if __name__ == "__main__":
    train()