from pathlib import Path

import hydra
from torch.utils.data import Dataset, DataLoader
import shutil
import csv
import pandas as pd
import torch
import os
from transformers import AutoTokenizer

from loguru import logger
from sklearn.model_selection import train_test_split

class DatasetManager(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path, output_data_path) -> None:
        # Download latest version
        self.raw_data_path = Path(raw_data_path)
        self.output_data_path = Path(output_data_path)
        self.data_path = ""

    def __len__(self) -> int:
        """Return the length of the dataset."""
        with open(self.data_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            return sum(1 for _ in reader)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        with open(self.data_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == index:
                    return row

    def preprocess(self, version : str = "latest") -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Preprocessing data...")
        # Implement the preprocessing logic here
        # Save the preprocessed data to the output folder
        for item in self.raw_data_path.iterdir():
            if item.is_file():
                shutil.copy(item, self.output_data_path / item.name)

        self.set_version(version)

    def set_version(self, version: str = "latest") -> None:
        """Set the version of the dataset."""
        highest_number = -1
        highest_file = None
        for item in self.output_data_path.iterdir():
            if item.is_file():
                try:
                    number = int(item.stem.split("_")[-1])
                    if number > highest_number:
                        highest_number = number
                        highest_file = item
                    if version != "latest":
                        if version == item.stem:
                            self.data_path = self.output_data_path / item.name
                            return
                except ValueError:
                    continue

        if highest_file:
            print(f"File with the highest number in the name: {highest_file}")
        else:
            print("No files found in the output folder")
            return

        self.data_path = self.output_data_path / highest_file.name


    def create_dataloader_from_set(self, tokenizer, cfg):
        """Create a PyTorch DataLoader from the dataset."""
        # Implement the logic to create a DataLoader

        data = pd.read_csv(self.data_path)
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

        train_dataset = LLMDataset(train_texts, train_labels, tokenizer, max_length=cfg.data.max_length)
        val_dataset = LLMDataset(val_texts, val_labels, tokenizer, max_length=cfg.data.max_length)
        test_dataset = LLMDataset(test_texts, test_labels, tokenizer, max_length=cfg.data.max_length)

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
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader


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

@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def preprocess(cfg) -> None:
    print("Creating an instance of dataset")
    dataset = DatasetManager(cfg.data.raw_data_path, cfg.data.processed_data_path)
    dataset.preprocess("lastest")

if __name__ == "__main__":
    preprocess()
