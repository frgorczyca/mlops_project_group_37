import torch
import hydra
from torch import nn
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer
from torch.optim import Adam, AdamW, SGD
from torchmetrics import Accuracy

from loguru import logger


class LLMDetector(pl.LightningModule):
    """
    PyTorch Lightning version of LLM Detector.
    """

    def __init__(self, cfg):
        super().__init__()
        logger.info("Initializing LLMDetector model")

        # Save hyperparameters to the checkpoint
        # self.save_hyperparameters(cfg)

        model_name = cfg.model.model_name
        logger.info(f"Loading transformer model: {model_name}")
        try:
            self.transformer = AutoModel.from_pretrained(model_name)
            logger.debug(f"Transformer config: {self.transformer.config}")
        except:
            logger.error(f"Unsupported model: {model_name}")
            raise ValueError(f"Model {model_name} not supported")

        self.dropout = nn.Dropout(cfg.model.dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, cfg.model.num_classes)
        logger.info(f"Initialized classifier with {cfg.model.num_classes} classes")

        # Initialize metrics with compute_on_step=False for better performance
        logger.debug("Initializing metrics")
        self.train_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)

        # Store training parameters
        self.optimizer_name = cfg.optimizer.type
        self.lr = cfg.optimizer.lr
        logger.info(f"Using optimizer: {self.optimizer_name} with learning rate: {self.lr}")

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.training.get('label_smoothing', 0.0)
        )
        logger.debug(f"Using CrossEntropyLoss with label_smoothing={cfg.training.get('label_smoothing', 0.0)}")

    def forward(self, input_ids, attention_mask):
        """Optimized forward pass"""
        try:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            pooled_output = outputs[0][:, 0, :]
            pooled_output = self.dropout(pooled_output)
            return self.classifier(pooled_output)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def _shared_step(self, batch, batch_idx, step_type='train'):
        """Shared step for train/val/test to reduce code duplication"""
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = self(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = getattr(self, f'{step_type}_accuracy')(preds, labels)

            # Log metrics
            self.log(f'{step_type}_loss', loss, on_step=(step_type=='train'), on_epoch=True, prog_bar=True)
            self.log(f'{step_type}_acc', acc, on_step=(step_type=='train'), on_epoch=True, prog_bar=True)

            if step_type == 'train' and batch_idx % 100 == 0:
                # logger.debug(f"Batch {batch_idx}: {step_type}_loss={loss:.4f}, {step_type}_acc={acc:.4f}")
                ...

            return loss
        except Exception as e:
            logger.error(f"Error in {step_type}_step: {str(e)}")
            raise e

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')


    def configure_optimizers(self):
        """Configure optimizer based on config file"""
        logger.info(f"Configuring {self.optimizer_name} optimizer")
        try:
            if self.optimizer_name == 'adam':
                optimizer = Adam(self.parameters(), lr=self.lr)
            elif self.optimizer_name == 'adamw':
                optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
            elif self.optimizer_name == 'sgd':
                optimizer = SGD(self.parameters(), lr=self.lr)
            else:
                logger.error(f"Unsupported optimizer: {self.optimizer_name}")
                raise ValueError(f"Optimizer {self.optimizer_name} not supported")
                
            return optimizer
        except Exception as e:
            logger.error(f"Failed to configure optimizer: {str(e)}")
            raise

    def predict_step(self, batch, batch_idx):
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            return torch.argmax(self(input_ids, attention_mask), dim=1)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


@hydra.main(version_base="1.1", config_path="../../configs", config_name="config")
def main(cfg):
    logger.info("Starting model testing")

    # Create tokenizer and model
    logger.info(f"Loading tokenizer and model: {cfg.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    model = LLMDetector(cfg)

    # Model summary with parameter count by layer
    logger.info("Model architecture summary:")
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}, of which trainable: {trainable_params:,}")

    # Test with dummy input
    logger.info("Testing model with dummy input")
    dummy_text = "This is a sample text to test the model."
    encoding = tokenizer.encode_plus(
        dummy_text,
        add_special_tokens=True, # Adds special tokens like [CLS] at start and [SEP] at end to help model understand input
        # max_length=cfg.data.max_length, # Maximum length of the sequence, if text is longer it will be truncated
        # padding='max_length', # Pads sequences to reach max_length
        # truncation=True, # If text is longer than max_length, it will be cut off
        return_attention_mask=True, # Returns a mask identifying real tokens vs padding
        return_tensors='pt' # Returns PyTorch tensors
    )

    with torch.no_grad():  # Add no_grad for inference
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        logger.debug(f"Input IDs shape: {input_ids.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")

        output = model(input_ids, attention_mask)
        logger.debug(f"Output shape: {output.shape}")
        logger.debug(f"Output (logits): {output}")

        prediction = torch.argmax(output, dim=1)
        logger.info(f"Prediction: {prediction.item()} (0: Human, 1: AI)")
        logger.info("Model testing completed")

if __name__ == "__main__":
    main()
