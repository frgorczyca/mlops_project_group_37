import logging
import operator
import os

import typer
import wandb
from dotenv import load_dotenv

from loguru import logger
load_dotenv()


logger.add(
    "logs/auto_register_best_model.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)


def stage_best_model_to_registry(artifact_name: str, metric_name: str = "best_val_accuracy", higher_is_better: bool = True) -> None:
    """
    Stage the best model to the model registry.

    Args:
        model_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.

    """
    api_key = os.getenv("WANDB_API_KEY")
    entity_name = os.getenv("WANDB_ENTITY")
    team_name = os.getenv("WANDB_TEAM")
    project_name = os.getenv("WANDB_PROJECT")
    registry_name = "Text detect registry"
    collection_name = "Trained LLM detector"

    artifact_name_path = f"{team_name}/{project_name}/{artifact_name}"
    artifact_registry_path = f"{entity_name}/wandb-registry-{registry_name}/{collection_name}"

    api = wandb.Api(api_key=api_key)

    artifact_collection = api.artifact_collection(type_name="model", name=artifact_name_path)

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        logger.info(f"Checking artifact: {artifact.name} with {metric_name}={artifact.metadata.get(metric_name)}")
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logging.error("No model found in registry.")
        return

    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    best_artifact.link(
        target_path=artifact_registry_path,
        aliases=["best", "staging"],
    )
    best_artifact.save()
    logger.info("Model staged to registry.")


if __name__ == "__main__":
    # typer.run(stage_best_model_to_registry)
    artifact_name = "llm-detector-model"
    stage_best_model_to_registry(artifact_name)