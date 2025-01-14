from torch.utils.data import Dataset

from text_detect import data


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = data.DatasetManager("data/raw")
    assert isinstance(dataset, Dataset)
