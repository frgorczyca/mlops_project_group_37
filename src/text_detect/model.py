import torch
import hydra
from torch import nn
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer
from torch.optim import Adam, AdamW, SGD
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
import wandb


class LLMDetector(pl.LightningModule):
    """
    PyTorch Lightning version of LLM Detector.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        logger.info("Initializing LLMDetector model")

        model_name = cfg.model.transformer_name
        logger.info(f"Loading transformer model: {model_name}")
        try:
            self.transformer = AutoModel.from_pretrained(model_name)
            logger.debug(f"Transformer config: {self.transformer.config}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            logger.error(f"Unsupported model: {model_name}")
            raise ValueError(f"Model {model_name} not supported")

        self.dropout = nn.Dropout(cfg.model.dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, cfg.data.num_classes)
        logger.info(f"Initialized classifier with {cfg.data.num_classes} classes")

        # Store training parameters
        self.optimizer_name = cfg.optimizer.type
        self.lr = cfg.optimizer.lr
        if self.optimizer_name == 'adamw':
            self.weight_decay = cfg.optimizer.weight_decay
            logger.info(f"Using optimizer: {self.optimizer_name} with learning rate: {self.lr} and weight decay: {self.weight_decay}")
        elif self.optimizer_name == 'sgd':
            self.momentum = cfg.optimizer.momentum
            logger.info(f"Using optimizer: {self.optimizer_name} with learning rate: {self.lr} and momentum: {self.momentum}")
        else:
            logger.info(f"Using optimizer: {self.optimizer_name} with learning rate: {self.lr}")

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.get("label_smoothing", 0.0))
        logger.debug(f"Using CrossEntropyLoss with label_smoothing={cfg.training.get('label_smoothing', 0.0)}")

        # Initialize metrics
        logger.debug("Initializing metrics")
        task_type = "binary" if cfg.data.num_classes == 2 else "multiclass"
        
        # Define metric kwargs based on task type
        metric_kwargs = {
            "task": task_type,
            "num_classes": cfg.data.num_classes,
            "average": "macro" if task_type == "multiclass" else None
        }
        
        # Initialize base metrics dictionary
        base_metrics = {
            'accuracy': Accuracy(**metric_kwargs),
        }

        # Add optional metrics based on configuration flags
        if cfg.logging.precision:
            base_metrics['precision'] = Precision(**metric_kwargs)
        if cfg.logging.recall:
            base_metrics['recall'] = Recall(**metric_kwargs)
        if cfg.logging.f1:
            base_metrics['f1'] = F1Score(**metric_kwargs)

        # Add AUROC separately since it needs probabilities
        if cfg.logging.auroc:
            auroc_kwargs = metric_kwargs.copy()
            if task_type == "binary":
                auroc_kwargs["task"] = "binary"
            base_metrics['auroc'] = AUROC(**auroc_kwargs)

        # Create MetricCollection
        metrics = MetricCollection(base_metrics)

        # Clone metrics for each stage with prefixes
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Initialize lists for storing outputs
        self.val_outputs = []
        self.test_outputs = []

        # Save class names for logging
        self.class_names = ["Human", "AI"] if cfg.data.num_classes == 2 else [f"Class_{i}" for i in range(cfg.data.num_classes)]

    def forward(self, input_ids, attention_mask):
        """Optimized forward pass"""
        try:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            pooled_output = outputs[0][:, 0, :]
            pooled_output = self.dropout(pooled_output)
            return self.classifier(pooled_output)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise

    def configure_optimizers(self):
        """Configure optimizer based on config file"""
        logger.info(f"Configuring {self.optimizer_name} optimizer")
        try:
            if self.optimizer_name == 'adam':
                optimizer = Adam(self.parameters(), lr=self.lr)
            elif self.optimizer_name == 'adamw':
                optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer_name == 'sgd':
                optimizer = SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
            else:
                logger.error(f"Unsupported optimizer: {self.optimizer_name}")
                raise ValueError(f"Optimizer {self.optimizer_name} not supported")

            return optimizer
        except Exception as e:
            logger.error(f"Failed to configure optimizer: {str(e)}")
            raise

    def _shared_step(self, batch, batch_idx, step_type):
        """Shared step with corrected metric computation"""
        try:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            logits = self(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Compute all metrics at once using MetricCollection
            metrics = getattr(self, f"{step_type}_metrics")
            
            # For AUROC, we need probabilities, for others we need predictions
            metric_dict = {}
            for name, metric in metrics.items():
                if name == f"{step_type}_auroc":
                    # AUROC expects probabilities
                    metric_dict[name] = metric(probs, labels)
                else:
                    # Other metrics expect class predictions
                    metric_dict[name] = metric(preds, labels)
            
            # Log metrics efficiently
            self.log(f"{step_type}_loss", loss, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)
            self.log_dict(metric_dict, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)

            # Store outputs for epoch-end visualization
            if step_type in ["val", "test"]:
                getattr(self, f"{step_type}_outputs").append({
                    "preds": preds,
                    "labels": labels,
                    "probs": probs
                })

            return loss
        except Exception as e:
            logger.error(f"Error in {step_type}_step: {str(e)}")
            raise e

    def on_validation_epoch_start(self):
        """Reset validation outputs at the start of each epoch"""
        self.val_outputs = []

    def on_test_epoch_start(self):
        """Reset test outputs at the start of each epoch"""
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric("val_accuracy", summary="max")
        return self._shared_step(batch, batch_idx, step_type="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, step_type="test")

    def predict_step(self, batch, batch_idx):
        try:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            return torch.argmax(self(input_ids, attention_mask), dim=1)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _shared_epoch_end(self, outputs, stage):
        """Shared epoch end logic for validation and test"""
        if not isinstance(self.logger, WandbLogger):
            return
            
        all_preds = torch.cat([x["preds"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])
        all_probs = torch.cat([x["probs"] for x in outputs])
        
        # Log rich visualizations to WandB
        self.logger.experiment.log({
            f"{stage}_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels.cpu().numpy(),
                preds=all_preds.cpu().numpy(),
                class_names=self.class_names
            ),
            f"{stage}_pr_curve": wandb.plot.pr_curve(
                all_labels.cpu().numpy(),
                all_probs.cpu().numpy(),
                labels=self.class_names
            ),
            f"{stage}_roc_curve": wandb.plot.roc_curve(
                all_labels.cpu().numpy(),
                all_probs.cpu().numpy(),
                labels=self.class_names
            )
        })
        
        # Clear outputs
        outputs.clear()

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.val_outputs, "val")

    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_outputs, "test")


@hydra.main(version_base="1.1", config_path="../../configs", config_name="default")
def main(cfg):
    logger.info("Starting model testing")

    # Create tokenizer and model
    logger.info(f"Loading tokenizer and model: {cfg.model.transformer_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)
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
        add_special_tokens=True,  # Adds special tokens like [CLS] at start and [SEP] at end to help model understand input
        # max_length=cfg.data.max_length, # Maximum length of the sequence, if text is longer it will be truncated
        # padding='max_length', # Pads sequences to reach max_length
        # truncation=True, # If text is longer than max_length, it will be cut off
        return_attention_mask=True,  # Returns a mask identifying real tokens vs padding
        return_tensors="pt",  # Returns PyTorch tensors
    )

    with torch.no_grad():  # Add no_grad for inference
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

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
