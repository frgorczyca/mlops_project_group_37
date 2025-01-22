from torch.utils.data import Dataset

from text_detect import data
import pytest
from pathlib import Path
import os
import pdb


@pytest.mark.parametrize("dataset_path", ["data/raw_data.csv"])
def test_my_dataset(dataset_path):
    """Test the MyDataset class."""
    dataset = data.DatasetManager(dataset_path)
    assert isinstance(dataset, Dataset)


@pytest.mark.parametrize("dataset_path", ["src/tests/mock_trainset_1.csv"])
def test_get_item(dataset_path):
    """Test the get_item property of dataset."""
    dataset = data.DatasetManager(dataset_path)
    item = dataset[1]
    assert item == ["1", "carrot"]


@pytest.mark.parametrize("dataset_path", ["src/tests/mock_trainset_1.csv"])
def test_dataset_length(dataset_path):
    """Test the len of dataset."""
    dataset = data.DatasetManager(dataset_path)
    assert len(dataset) == 17


@pytest.fixture
def setup_test_environment(tmp_path):
    # Create a temporary input file
    input_folder = tmp_path / "input_folder"
    input_folder.mkdir()
    input_file = input_folder / "mock_trainset_1.csv"
    input_file.write_text("sample data")

    # Create a temporary output folder
    output_folder = tmp_path / "output_folder"
    output_folder.mkdir()

    return input_folder, output_folder, input_file

def test_dataset_preprocess(setup_test_environment):
    """Test method preprocess"""
    input_folder, output_folder, input_file = setup_test_environment

    # Create an instance of DatasetManager
    dataset = data.DatasetManager(Path(input_folder))

    # Call the preprocess method
    dataset.preprocess(output_folder=Path(output_folder))

    # Assert that the file has been copied
    copied_file = output_folder / input_file.name
    assert copied_file.exists()
    assert copied_file.read_text() == "sample data"
    
## only tests w version="latest". Not sure if we need to test for other versions?
def test_dataset_get_version(setup_test_environment):
    input_folder, output_folder, input_file = setup_test_environment
    dataset = data.DatasetManager(Path(input_folder))
    dataset.preprocess(output_folder=Path(output_folder))
    dataset.get_version(Path(output_folder))
    assert dataset.data_path == output_folder / input_file.name
    
