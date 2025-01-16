from pathlib import Path

import typer
from torch.utils.data import Dataset
import shutil
import csv
import pandas as pd
import numpy as np
import os

#Read 
def load_csv_to_dataframe(file_path: Path):
    try:
        df=pd.read_csv(file_path)
        print("DataFrame loaded successfully. Here is a preview: ")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

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
    def get_version(self, input_folder: Path, version: int | str = "latest") -> None:
        """Get a specific version of the dataset: either int or latest"""
        if isinstance(version, int):
            version_str=str(version)
            for file_name in os.listdir(input_folder):
                if version_str in file_name:
                    return os.path.join(input_folder, file_name)
            
        
        elif version == "latest":
            highest_number = -1
            highest_file = None
            for item in input_folder.iterdir():
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
                return highest_file
            else:
                print("No files found in the output folder")
                return
            
            self.data_path = input_folder / highest_file.name
        else:
            self.data_path = input_folder / f"data_{version}.csv"
            
    def preprocess(self, version, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Preprocessing data...")
        shutil.copy(version, output_folder )

        self.get_version(output_folder)




def preprocess(raw_data_path: Path, output_folder: Path, version: str | int = "latest") -> None:
    print("Creating an instance of dataset")
    dataset = DatasetManager(raw_data_path)
    file_version = dataset.get_version(raw_data_path,version)
    dataset.preprocess(file_version, output_folder)
    
    print()


if __name__ == "__main__":
    #typer.run(preprocess)
    preprocess(Path("data/raw"), Path("data/processed"), 2)

load_csv_to_dataframe("C:/Users/arttu/Desktop/mlops/final_project/mlops_project_group_37/data/raw/train_drcat_04.csv")