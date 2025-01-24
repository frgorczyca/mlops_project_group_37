"""
Script to test the our data (data.py)
"""

from torch.utils.data import Dataset

from text_detect import data
from text_detect.data import LLMDataset
import pytest
from pathlib import Path
from hydra import initialize, compose
from transformers import AutoTokenizer
import torch

## Create our mock dataset for testing
@pytest.fixture(scope="module")
def dataset():
    with initialize(version_base="1.1", config_path="../../../configs"):
        # Compose the configuration
        cfg = compose(config_name="default")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)
        dummy_texts = ["This is a test"]
        dummy_labels = [0]
        dataset = LLMDataset(texts=dummy_texts, 
                             labels=dummy_labels, 
                             tokenizer=tokenizer, 
                             max_length=cfg.data.max_length)
        return dataset


#@pytest.mark.parametrize("dummy_texts, dummy_labels",[(["This is a test"], [0])])
def test_my_dataset(dataset) -> None:
    assert isinstance(dataset, LLMDataset)

def test_len(dataset):
    dataset_len = len(dataset)
    assert dataset_len==1

@pytest.mark.parametrize("idx", [0])
def test_get_item(dataset, idx):
    item = dataset[idx]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape == (dataset.max_length,)
    assert item["attention_mask"].shape == (dataset.max_length,)
    assert item["labels"].item() == dataset.labels[idx]

@pytest.mark.parametrize("idx", [0])
def test_get_text(dataset,idx):
    assert dataset.get_text(idx) == "This is a test"


@pytest.mark.parametrize("idx", [0])
def test_get_label(dataset, idx):
    assert dataset.get_label(idx) == 0


## Create mock data for testing
@pytest.fixture
def setup_test_environment(tmp_path):
    # Create a temporary input file
    input_folder = tmp_path / "input_folder"
    input_folder.mkdir()
    input_file1 = input_folder / "train_drcat_1.csv"
    input_file1.write_text("sample data")
    input_file2 = input_folder / "train_drcat_2.csv"
    input_file2.write_text("sample data")

    # Create a temporary output folder
    output_folder = tmp_path / "output_folder"
    output_folder.mkdir()
    output_file = output_folder / "output.csv"

    return input_folder, input_file1, input_file2, output_file


def test_load_latest_file(setup_test_environment) -> None:
    input_folder, input_file1, input_file2, output_file = setup_test_environment
    highest_file = data.load_latest_file(input_folder)
    assert input_file1.exists()
    assert highest_file == input_file2

@pytest.mark.parametrize("csv_file", ["src/tests/mock_trainset_1.csv"])
def test_load_data(csv_file) -> None:
    texts, labels = data.load_data(csv_file)
    assert texts[0] == "carrot", f"texts[0] is not 'carrot': {texts[0]}"
    assert labels[0] == 1, f"labels[0] is not 1: {labels[0]}"
    
@pytest.mark.parametrize("texts, labels" , [(["hej"], [0]), (["hej hej"], [1])])
def test_save_to_csv(texts, labels, setup_test_environment) -> None:
    input_folder, input_file1, input_file2, output_file = setup_test_environment
    data.save_to_csv(texts, labels, output_file)
    assert output_file.exists()
    assert output_file.read_text() == f"text,label\n{texts[0]},{labels[0]}\n"
    
    
    
    