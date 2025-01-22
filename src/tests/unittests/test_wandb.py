from text_detect.wandb_functions import load_wandb_env_vars
from dotenv import load_dotenv

load_dotenv()


def test_load_wandb_env_vars():
    """Test the load_wandb_env_vars function returns correct number of values."""
    result = load_wandb_env_vars()
    assert isinstance(result, tuple)
    assert len(result) == 6  # since it returns 6 values
