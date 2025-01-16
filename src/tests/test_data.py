from torch.utils.data import Dataset

from text_detect import data
import pytest


@pytest.mark.parametrize("dataset_path", ["data/raw_data.csv"])
def test_my_dataset(dataset_path):
    """Test the MyDataset class."""
    dataset = data.DatasetManager(dataset_path)
    assert isinstance(dataset, Dataset)


@pytest.mark.parametrize("dataset_path", ["src/tests/mock_trainset.csv"])
def test_get_item(dataset_path):
    """Test the get_item property of dataset."""
    dataset = data.DatasetManager(dataset_path)
    item = dataset[1]
    assert item == ["1", "carrot"]


@pytest.mark.parametrize("dataset_path", ["src/tests/mock_trainset.csv"])
def test_dataset_length(dataset_path):
    """Test the len of dataset."""
    dataset = data.DatasetManager(dataset_path)
    assert len(dataset) == 17
