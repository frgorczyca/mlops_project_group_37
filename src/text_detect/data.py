import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig


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
    
    def get_text(self, idx):
        return self.texts[idx]
    
    def get_label(self, idx):
        return self.labels[idx]


def load_latest_file(base_path: str) -> Path:
    """Retrieve the latest 'train_drcat_<number>.csv' file."""
    raw_data_path = Path(base_path)
    highest_number = -1
    highest_file = None
    for item in raw_data_path.iterdir():
        if item.is_file() and item.name.startswith("train_drcat_"):
            try:
                # Extract number from filename (e.g., 'train_drcat_04.csv' -> 4)
                number = int(item.stem.split("_")[-1])
                print(f"Found dataset: {item} with number {number}")
                if number > highest_number:
                    highest_number = number
                    highest_file = item
            except ValueError:
                continue
    
    if highest_file:
        print(f"Using latest dataset: {highest_file}")
        return highest_file
    else:
        raise FileNotFoundError("No suitable 'train_drcat' files found.")

def load_data(data_path: Path) -> (list, list):
    """Load data from the CSV and extract texts and labels."""
    texts, labels = [], []
    with open(data_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label']))
    return texts, labels

def save_to_csv(texts: list, labels: list, output_file: Path):
    """Save texts and labels to a CSV file."""
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    df.to_csv(output_file, index=False)

def preprocess_and_split(cfg: DictConfig, data_path: Path):
    """Preprocess, split the dataset, and save it to the processed folder."""
    # Load the raw data
    texts, labels = load_data(data_path)

    # Split the dataset into training and testing sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=cfg.data.test_size,
        random_state=cfg.seed
    )

    # Create processed folder if it doesn't exist
    output_folder = Path("data/processed")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    save_to_csv(train_texts, train_labels, output_folder / "train.csv")
    save_to_csv(test_texts, test_labels, output_folder / "test.csv")

@hydra.main(version_base="1.1", config_path="../../configs", config_name="default")
def preprocess(cfg: DictConfig) -> None:
    """Function to initialize dataset processing and preprocess data."""
    # Load the latest dataset file
    latest_file = load_latest_file("data/raw")
    
    # Preprocess and split the data
    preprocess_and_split(cfg, latest_file)

if __name__ == "__main__":
    preprocess()
