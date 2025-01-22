"""
Script to test the our model (model.py)
"""

import torch
from text_detect.model import LLMDetector


def test_model():
    model = LLMDetector()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
