import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from text_detect.wandb_functions import load_wandb_env_vars, load_download_artifact_model, cleanup_downloaded_model
from omegaconf import OmegaConf
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader, Dataset
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals for model, tokenizer, and config
model = None
tokenizer = None
cfg = None
trainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading and cleaning up the model."""
    try:
        logger.info("Starting application and loading model...")
        load_model()
        yield
    finally:
        logger.info("Shutting down application and cleaning up resources...")
        cleanup_model()


# Initialize FastAPI app with lifespan
app = FastAPI(title="Text Detection API", lifespan=lifespan)


# Pydantic models for request/response
class InferenceRequest(BaseModel):
    text: str


class InferenceResponse(BaseModel):
    predicted_class: str
    human_probability: float
    ai_probability: float
    raw_logits: list


class SinglePredictionDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[0], "attention_mask": self.attention_mask[0]}


def load_model():
    """Load model and tokenizer if not already loaded."""
    global model, tokenizer, cfg, trainer

    if model is not None:
        logger.info("Model is already loaded.")
        return

    try:
        logger.info("Loading production model.")
        production_model_artifact_path = (
            "mlops_project_group_37/wandb-registry-Text detect registry/Trained LLM detector:production"
        )

        api_key, _, _, _, _, _ = load_wandb_env_vars()
        api = wandb.Api(api_key=api_key)
        artifact = api.artifact(production_model_artifact_path)

        # Get wandb config
        wb_config = artifact.logged_by().config
        cfg = OmegaConf.create(wb_config)

        _, model = load_download_artifact_model(cfg, api, production_model_artifact_path)

        # Set random seeds
        pl.seed_everything(cfg.seed)

        # Load tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.transformer_name)

        # Initialize trainer
        trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

        logger.info("Model and tokenizer loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Error loading model: {e}")


def cleanup_model():
    """Cleanup model resources."""
    global model, tokenizer, cfg, trainer

    try:
        if model is not None:
            logger.info("Cleaning up model resources...")

            # Clean up PyTorch model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clean up downloaded model files
            cleanup_downloaded_model()

            model = None
            tokenizer = None
            cfg = None
            trainer = None

            logger.info("Model resources cleaned up.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Endpoint for text classification.

    Args:
        request: InferenceRequest object containing text

    Returns:
        InferenceResponse object with prediction results
    """
    try:
        # Ensure model is loaded
        if model is None:
            logger.warning("Model not loaded. Loading now...")
            load_model()

        # Tokenize input
        encoded_input = tokenizer.encode_plus(
            request.text,
            add_special_tokens=True,
            max_length=cfg.data.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Create dataset and dataloader
        dataset = SinglePredictionDataset(encoded_input["input_ids"], encoded_input["attention_mask"])
        predict_dataloader = DataLoader(dataset, batch_size=1)

        # Run inference
        predictions = trainer.predict(model, predict_dataloader)

        # Process results
        logits = predictions[0].squeeze()
        probabilities = torch.softmax(logits, dim=-1)
        class_names = ["Human", "AI"]

        predicted_class = logits.argmax(-1).item()
        probs_array = probabilities.detach().cpu().numpy()

        logger.info(
            f"Predicted class: {predicted_class} ({class_names[predicted_class]}), "
            f"Input text {request.text[:100]}{'...' if len(request.text) >= 100 else ''}"
        )

        return InferenceResponse(
            predicted_class=class_names[predicted_class],
            human_probability=float(probs_array[0]),
            ai_probability=float(probs_array[1]),
            raw_logits=logits.tolist(),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
