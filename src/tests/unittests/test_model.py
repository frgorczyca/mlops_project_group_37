# test_model.py
import pytest
import torch
import hydra
from omegaconf import DictConfig

# Adjust the import path to point to your model.py, e.g.:
# from my_project.model import LLMDetector
from text_detect.model import LLMDetector

from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    """
    A pytest fixture that initializes Hydra and composes the default config.
    """
    with hydra.initialize(config_path="../../../configs", version_base="1.1"):
        # 'default' is the config_name specified in your default.yaml
        composed_cfg = hydra.compose(config_name="default")
    return composed_cfg


@pytest.fixture(scope="session")
def detector(cfg: DictConfig) -> LLMDetector:
    """
    A pytest fixture that creates an LLMDetector instance
    using the real Hydra configuration.
    """
    model = LLMDetector(cfg)
    return model


@pytest.fixture(scope="session")
def tokenizer(cfg: DictConfig):
    """
    A pytest fixture that creates a tokenizer matching the transformer model
    in the config.
    """
    return AutoTokenizer.from_pretrained(cfg.model.transformer_name)


@pytest.fixture
def dummy_batch(cfg: DictConfig, tokenizer) -> dict:
    """
    Create a dummy batch dictionary compatible with the model's _shared_step method.
    The batch will contain 'input_ids', 'attention_mask', and 'labels'.
    """
    text = "This is a dummy sentence for testing."
    encoding = tokenizer.encode_plus(
        text,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # For classification, we need labels. Let's create a single label in [0..(num_classes-1)].
    # If num_classes=2, label=0 or 1. We'll pick 0 for this test.
    labels = torch.tensor([0])

    batch = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }
    return batch


def test_model_initialization(cfg, detector):
    """
    Test that the LLMDetector initializes with the correct config parameters.
    For instance, checks that number of classes matches, dropout, etc.
    """
    assert detector.cfg is not None, "Model did not store the cfg."
    assert detector.transformer is not None, "Transformer model not loaded."
    assert detector.classifier.out_features == cfg.data.num_classes, \
        "Classifier output does not match cfg.data.num_classes."


def test_forward_pass(detector, dummy_batch):
    """
    Test the forward method with a dummy batch.
    Check that it returns the correct shape: (batch_size, num_classes).
    """
    input_ids = dummy_batch["input_ids"]
    attention_mask = dummy_batch["attention_mask"]

    logits = detector(input_ids, attention_mask)
    assert logits.shape[0] == input_ids.shape[0], "Batch dimension mismatch."
    assert logits.shape[1] == detector.cfg.data.num_classes, "Num_classes dimension mismatch."


def test_configure_optimizers(cfg, detector):
    """
    Test that configure_optimizers returns an optimizer instance
    matching the specified optimizer type in the config.
    """
    optimizer = detector.configure_optimizers()

    opt_type = cfg.optimizer.type.lower()
    if opt_type == "adam":
        assert optimizer.__class__.__name__ == "Adam", \
            f"Expected Adam, got {optimizer.__class__.__name__}"
    elif opt_type == "adamw":
        assert optimizer.__class__.__name__ == "AdamW", \
            f"Expected AdamW, got {optimizer.__class__.__name__}"
    elif opt_type == "sgd":
        assert optimizer.__class__.__name__ == "SGD", \
            f"Expected SGD, got {optimizer.__class__.__name__}"
    else:
        pytest.fail(f"Unsupported optimizer type: {opt_type}")

    # Check that the optimizer has the correct LR
    assert optimizer.defaults["lr"] == pytest.approx(cfg.optimizer.lr), \
        f"Optimizer LR mismatch: expected {cfg.optimizer.lr}, got {optimizer.defaults['lr']}"


def test_training_step(detector, dummy_batch):
    """
    Test the training_step. We only verify that it runs without error
    and returns a valid scalar loss.
    """
    loss = detector.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "training_step should return a Tensor loss."
    assert loss.dim() == 0, "Loss should be a scalar (0-dim Tensor)."

@pytest.mark.skip(reason="Integration test example; remove or adjust if you want to test epoch end hooks.")
def test_validation_step(detector, dummy_batch):
    """
    Test the validation_step. Similar check to training_step.
    """
    loss = detector.validation_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "validation_step should return a Tensor loss."
    assert loss.dim() == 0, "Loss should be a scalar (0-dim Tensor)."


def test_test_step(detector, dummy_batch):
    """
    Test the test_step. Similar check to training_step.
    """
    loss = detector.test_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "test_step should return a Tensor loss."
    assert loss.dim() == 0, "Loss should be a scalar (0-dim Tensor)."


def test_predict_step(detector, tokenizer):
    """
    Test the predict_step by passing a small input batch
    and checking the shape of the returned logits.
    """
    text = "Another dummy text for prediction."
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    batch = (
        encoding["input_ids"],
        encoding["attention_mask"],
    )

    # predict_step expects a tuple of (input_ids, attention_mask)
    logits = detector.predict_step(batch, batch_idx=0)
    assert logits.shape[0] == 1, "Expected a single sample in the batch."
    assert logits.shape[1] == detector.cfg.data.num_classes, "Num_classes dimension mismatch."


@pytest.mark.skip(reason="Integration test example; remove or adjust if you want to test epoch end hooks.")
def test_on_validation_epoch_end(detector):
    """
    Example test for on_validation_epoch_end.
    Typically, you'd want to test the confusion matrix logging,
    but that requires a real W&B run context or mocking.
    """
    # We won't do anything complicated here. Just call the hook to ensure it doesn't error out.
    detector.on_validation_epoch_end()
    # If no exception is raised, we consider it passed.


@pytest.mark.skip(reason="Integration test example; remove or adjust if you want to test epoch end hooks.")
def test_on_test_epoch_end(detector):
    """
    Example test for on_test_epoch_end.
    Again, typically you'd mock W&B calls.
    """
    detector.on_test_epoch_end()
    # If no exception is raised, we consider it passed.

