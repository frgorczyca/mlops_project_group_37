import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import hydra
import os
import wandb

from text_detect.model import LLMDetector
from text_detect.data import LLMDataset, load_data


def load_model(cfg, artifact_name, version):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_TEAM"), "project": os.getenv("WANDB_PROJECT")},
    )

    artifact_project_path = f"{artifact_name}:{version}"

    artifact = api.artifact(artifact_project_path)
    artifact.download(root="models/downloads")
    file_name = artifact.files()[0].name
    model_path = os.path.join("models/downloads", file_name)
    return LLMDetector.load_from_checkpoint(model_path, cfg=cfg)


@hydra.main(version_base="1.1", config_path="../../configs", config_name="default")
def evaluate(cfg):
    # Load the model from the checkpoint
    model = load_model(cfg, cfg.evaluate.artifact_name, cfg.evaluate.version)

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.model.transformer_name}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set this environment variable before importing tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)

    test_texts, test_labels = load_data(cfg.data.test_path)
    test_dataset = LLMDataset(texts=test_texts, labels=test_labels, tokenizer=tokenizer, max_length=cfg.data.max_length)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    test_results = trainer.test(model=model, dataloaders=test_loader)
    print({"test_loss": test_results[0]["test_loss"], "test_accuracy": test_results[0]["test_accuracy"]})

if __name__ == "__main__":
    evaluate()
