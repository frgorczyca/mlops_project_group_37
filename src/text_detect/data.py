from pathlib import Path

import typer
import kagglehub as kh
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        # Download latest version
        raw_data_path = kh.dataset_download("thedrcat/daigt-proper-train-dataset")
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

#print("Path to dataset files:", path)


if __name__ == "__main__":
    #typer.run(preprocess)
    dataset = MyDataset("data")
