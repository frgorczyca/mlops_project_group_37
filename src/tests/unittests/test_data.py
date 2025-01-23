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
    




# @pytest.mark.parametrize("dataset_path", ["src/tests/mock_trainset_1.csv"])
# def test_dataset_length(dataset_path):
#     """Test the len of dataset."""
#     dataset = LLMDataset(dataset_path)
#     assert len(dataset) == 17


# @pytest.fixture
# def setup_test_environment(tmp_path):
#     # Create a temporary input file
#     input_folder = tmp_path / "input_folder"
#     input_folder.mkdir()
#     input_file = input_folder / "mock_trainset_1.csv"
#     input_file.write_text("sample data")

#     # Create a temporary output folder
#     output_folder = tmp_path / "output_folder"
#     output_folder.mkdir()

#     return input_folder, output_folder, input_file


# def test_dataset_preprocess(setup_test_environment):
#     """Test method preprocess"""
#     input_folder, output_folder, input_file = setup_test_environment

#     # Create an instance of DatasetManager
#     dataset = data.DatasetManager(Path(input_folder))

#     # Call the preprocess method
#     dataset.preprocess(output_folder=Path(output_folder))

#     # Assert that the file has been copied
#     copied_file = output_folder / input_file.name
#     assert copied_file.exists()
#     assert copied_file.read_text() == "sample data"


## only tests w version="latest". Not sure if we need to test for other versions?
# def test_dataset_get_version(setup_test_environment):
#     """Test method get_version"""
#     input_folder, output_folder, input_file = setup_test_environment
#     dataset = data.DatasetManager(Path(input_folder))
#     dataset.preprocess(output_folder=Path(output_folder))
#     dataset.get_version(Path(output_folder))
#     assert dataset.data_path == output_folder / input_file.name







# _________


# with initialize(version_base=None, config_path="../../../configs/"):
#         cfg = compose(config_name="default")
#         assert cfg.num_classes == 2, f"num_classes is not 2: {cfg.num_classes}"
#         assert 0 <= cfg.test_size <= 1, f"test_size is out of range: {cfg.test_size}"
#         assert cfg.max_length > 0, f"max_length is not positive: {cfg.max_length}"
#         train_path = Path(cfg.train_path)
#         test_path = Path(cfg.test_path)
#         assert train_path.exists(), f"train_path does not exist: {cfg.train_path}"
#         assert test_path.exists(), f"test_path does not exist: {cfg.train_path}"
#         test_texts, test_labels = data.load_data(cfg.train_path)