from pathlib import Path

import typer
from torch.utils.data import Dataset
import shutil
import csv

class DatasetManager(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        # Download latest version
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""
        with open(self.data_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            return sum(1 for _ in reader)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        with open(self.data_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == index:
                    return row

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Preprocessing data...")
        # Implement the preprocessing logic here
        # Save the preprocessed data to the output folder
        for item in self.data_path.iterdir():
            if item.is_file():
                shutil.copy(item, output_folder / item.name)

        self.get_version(output_folder)

    def get_version(self, output_folder: Path, version: str = "latest") -> None:
        if version == "latest":
            highest_number = -1
            highest_file = None
            for item in output_folder.iterdir():
                if item.is_file():
                    try:
                        number = int(item.stem.split('_')[-1])
                        if number > highest_number:
                            highest_number = number
                            highest_file = item
                    except ValueError:
                        continue

            if highest_file:
                print(f"File with the highest number in the name: {highest_file}")
            else:
                print("No files found in the output folder")
                return

            self.data_path = output_folder / highest_file.name
        else:
            self.data_path = output_folder / f"data_{version}.csv"


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Creating an instance of dataset")
    dataset = DatasetManager(raw_data_path)
    dataset.preprocess(output_folder)
    dataset.get_version(output_folder)


if __name__ == "__main__":
    #typer.run(preprocess)
    preprocess(Path("data/raw"), Path("data/processed"))
