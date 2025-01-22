import bentoml
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import wandb
import os
from text_detect.wandb_functions import load_download_artifact_model
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 10})
class TextDetectionService:
    # tokenizer_service = bentoml.depends(TextTokenizerService)

    def __init__(self):
        logger.debug("Initializing TextDetectionService")

        # Load model configuration and weights
        self.production_model_artifact_path = (
            "mlops_project_group_37/wandb-registry-Text detect registry/Trained LLM detector:production"
        )
        api_key = api_key = os.getenv("WANDB_API_KEY")
        api = wandb.Api(api_key=api_key)
        artifact = api.artifact(self.production_model_artifact_path)
        wb_config = artifact.logged_by().config
        self.cfg = OmegaConf.create(wb_config)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.transformer_name)

        # Load model
        _, self.model = load_download_artifact_model(self.cfg, api, self.production_model_artifact_path)
        self.trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
        self.class_names = ["Human", "AI"]

    @bentoml.api(
        batchable=True,
        batch_dim=0,
    )
    def predict(self, input_texts: list[str]) -> list[str]:
        logger.info(f"Processing prediction request for {len(input_texts)} texts")
        try:
            # Log sample inputs
            logger.debug("Input text samples:")
            for i, text in enumerate(input_texts[:2]):
                logger.debug(f"Text {i}: {text[:100]}...")

            # Get tokenized inputs
            logger.debug("Requesting tokenization")

            encoded_inputs = self.tokenizer.batch_encode_plus(
                input_texts,
                add_special_tokens=True,
                max_length=self.cfg.data.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            logger.debug(f"Encoded input IDs shape: {encoded_inputs['input_ids'].shape}")
            logger.debug(f"Encoded attention mask shape: {encoded_inputs['attention_mask'].shape}")

            dataloader = DataLoader(
                TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"]),
                batch_size=len(encoded_inputs["input_ids"]),  # Process all inputs in one batch
                shuffle=False,
            )

            logger.debug("Running model inference using PyTorch Lightning trainer")
            # Use trainer.predict which handles device management automatically
            logits = self.trainer.predict(self.model, dataloader)
            logger.info(f"Logits: {logits}")
            predictions = torch.softmax(logits[0], dim=-1)
            logger.info(f"Predictions {predictions}")
            # Get the index of the maximum probability for each prediction
            pred_indices = predictions.argmax(dim=1).tolist()
            logger.info(f"Pred indices: {pred_indices}")
            # Convert to class names using the indices
            pred_classes = [self.class_names[idx] for idx in pred_indices]
            logger.info(f"Prediction complete. Predicted classes: {pred_classes}")
            return pred_classes

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise e
