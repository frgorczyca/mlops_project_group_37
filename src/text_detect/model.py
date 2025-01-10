import torch
import hydra
from torch import nn
import pytorch_lightning as pl
from transformers import RobertaModel, RobertaTokenizer
from torch.optim import Adam, AdamW, SGD
from torchmetrics import Accuracy
from typing import Dict, Any


class LLMDetector(pl.LightningModule):
    """
    PyTorch Lightning version of LLM Detector.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # Save hyperparameters to the checkpoint
        self.save_hyperparameters(cfg)

        # Initialize RoBERTa with gradient checkpointing
        self.roberta = RobertaModel.from_pretrained(cfg.model.model_name)

        self.dropout = nn.Dropout(cfg.model.dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, cfg.model.num_classes)
    
        # Initialize metrics with compute_on_step=False for better performance
        self.train_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)

        # Store training parameters
        self.lr = cfg.optimizer.lr

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.training.get('label_smoothing', 0.0)
        )

    def forward(self, input_ids, attention_mask):
        """Optimized forward pass"""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False  # Disable unused output
        )
        pooled_output = outputs[0][:, 0, :]  # Use CLS token output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def _shared_step(self, batch, batch_idx, step_type='train'):
        """Shared step for train/val/test to reduce code duplication"""
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
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        """Configure optimizer based on config file"""
        optimizer_name = self.cfg.optimizer.get('type', 'Adam').lower()
        
        if optimizer_name == 'adam':
            optimizer = Adam(self.parameters(), lr=self.lr)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.lr)
        elif optimizer_name == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
            
        return optimizer

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        return torch.argmax(self(input_ids, attention_mask), dim=1)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    # Create tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model.model_name)
    model = LLMDetector(cfg)
    
    # Model summary with parameter count by layer
    print(f"Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with dummy input
    dummy_text = "This is a sample text to test the model."
    encoding = tokenizer.encode_plus(
        dummy_text,
        add_special_tokens=True,
        # max_length=cfg.data.max_length,
        # padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():  # Add no_grad for inference
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        print(f"\nInput shapes:")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        output = model(input_ids, attention_mask)
        print(f"Output shape: {output.shape}")
        print(f"Output (logits): {output}")
        
        prediction = torch.argmax(output, dim=1)
        print(f"\nPrediction: {prediction.item()} (0: Human, 1: AI)")

if __name__ == "__main__":
    main()