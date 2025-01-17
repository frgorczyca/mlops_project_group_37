import torch
import hydra
from torch import nn
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer
from torch.optim import Adam, AdamW, SGD
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
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

        # Save hyperparameters to the checkpoint
        # self.save_hyperparameters(cfg)

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

        # Initialize metrics with compute_on_step=False for better performance
        logger.debug("Initializing metrics")
        self.train_accuracy = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)

        #Additional metrics to track: precision, recall, F1 Score and AUROC 
        self.train_precision = Precision(task="multiclass", num_classes=cfg.data.num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=cfg.data.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=cfg.data.num_classes)

        self.val_precision = Precision(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=cfg.data.num_classes)
        
        self.test_precision = Precision(task="binary" or "multiclass", num_classes=cfg.data.num_classes)
        self.test_recall = Recall(task="binary" or "multiclass", num_classes=cfg.data.num_classes)
        self.test_f1 = F1Score(task="binary" or "multiclass", num_classes=cfg.data.num_classes)
        self.test_accuracy = Accuracy(task="binary" or "multiclass", num_classes=cfg.data.num_classes)
        
        # AUROC for binary classification: set task="binary" if your labels are [0, 1].
        # If "multiclass" with 2 classes, specify that.
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        # Add confusion matrix metric
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=cfg.data.num_classes)
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=cfg.data.num_classes)
        
        # Lists to store predictions and labels for epoch-end logging
        self.val_step_outputs = []
        self.test_step_outputs = []

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
        """Shared step for train/val/test to reduce code duplication"""
        try:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            logits = self(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = getattr(self, f"{step_type}_accuracy")(preds, labels)
            prec = getattr(self, f"{step_type}_precision")(preds, labels)
            rec = getattr(self, f"{step_type}_recall")(preds, labels)
            f1 = getattr(self, f"{step_type}_f1")(preds, labels)
            
            # AUROC typically requires raw probabilities or logits, so pass logits or softmax output
            # For binary classification, you might do:
            if self.cfg.data.num_classes == 2:
                proba = torch.softmax(logits, dim=1)[:, 1]  # shape: (N,)
                auroc_val = getattr(self, f"{step_type}_auroc")(proba, labels)
            else:
                # for multiclass, pass the entire logits or softmax
                proba = torch.softmax(logits, dim=1)
                auroc_val = getattr(self, f"{step_type}_auroc")(proba, labels)

            self.log(f"{step_type}_acc", acc, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_loss", loss, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_prec", prec, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_rec", rec, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_f1", f1, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)
            self.log(f"{step_type}_auroc", auroc_val, on_step=(step_type=="train"), on_epoch=True, prog_bar=True)

            # Store predictions and labels for confusion matrix if not training
            if step_type in ["val", "test"]:
                output = {
                    "preds": preds,
                    "labels": labels,
                    "probs": torch.softmax(logits, dim=1)
                }
                if step_type == "val":
                    self.val_step_outputs.append(output)
                else:
                    self.test_step_outputs.append(output)

            return loss
        
        except Exception as e:
            logger.error(f"Error in {step_type}_step: {str(e)}")
            raise e

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
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

    def on_validation_epoch_end(self):
        """Log confusion matrix at the end of validation epoch"""
        if not isinstance(self.logger, WandbLogger):
            return
            
        # Concatenate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.val_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.val_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.val_step_outputs])
        
        # Calculate confusion matrix
        conf_mat = self.val_confmat(all_preds, all_labels)
        
        # Log to W&B
        self.logger.experiment.log({
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels.cpu().numpy(),
                preds=all_preds.cpu().numpy(),
                class_names=["Human", "AI"]  # Adjust based on your classes
            ),
            "val_pr_curve": wandb.plot.pr_curve(
                all_labels.cpu().numpy(),
                all_probs.cpu().numpy(),
                labels=["Human", "AI"]  # Adjust based on your classes
            )
        })
        
        # Clear saved outputs
        self.val_step_outputs.clear()

    def on_test_epoch_end(self):
        """Log confusion matrix at the end of test epoch"""
        if not isinstance(self.logger, WandbLogger):
            return
            
        # Concatenate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.test_step_outputs])
        
        # Calculate confusion matrix
        conf_mat = self.test_confmat(all_preds, all_labels)
        
        # Log to W&B
        self.logger.experiment.log({
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels.cpu().numpy(),
                preds=all_preds.cpu().numpy(),
                class_names=["Human", "AI"]  # Adjust based on your classes
            ),
            "test_pr_curve": wandb.plot.pr_curve(
                all_labels.cpu().numpy(),
                all_probs.cpu().numpy(),
                labels=["Human", "AI"]  # Adjust based on your classes
            )
        })
        
        # Clear saved outputs
        self.test_step_outputs.clear()


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
