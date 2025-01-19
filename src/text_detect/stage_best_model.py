import logging
import operator

import typer
import wandb
from dotenv import load_dotenv

from loguru import logger

from text_detect.link_model import load_wandb_env_vars

load_dotenv()


logger.add(
    "logs/stage_best_model.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


def stage_best_model_to_registry(
    artifact_name: str = typer.Argument(..., help="Name of the model to be registered"),
    artifact_type: str = typer.Option("model", "--type", "-t", help="Type of the artifact to be registered"),
    metric_name: str = typer.Option("best_val_accuracy", "--metric", "-m", help="Metric to choose the best model from"),
    higher_is_better: bool = typer.Option(
        True, "--higher-is-better/--lower-is-better", help="Whether higher metric values are better"
    ),
) -> None:
    """
    Stage the best model to the model registry based on the specified metric..

    Args:
        model_name: Name of the model to be registered.
        artifact_type: Type of the artifact to be registered, e.g. 'model'.
        metric_name: Name of the metric to choose the best model from, e.g. 'best_val_accuracy'.
        higher_is_better: Whether higher metric values are better, default is True.
    """
    # Load environment variables
    api_key, team_name, project_name, entity_name, registry_name, collection_name = load_wandb_env_vars()

    # Define artifact paths for linking between team project artifact and organization registry collection
    artifact_name_project_path = f"{team_name}/{project_name}/{artifact_name}"
    artifact_registry_path = f"{entity_name}/wandb-registry-{registry_name}/{collection_name}"

    # Initialize W&B API
    api = wandb.Api(api_key=api_key)

    # Load artifacts from team project
    logger.info(f"Loading artifacts from {artifact_name_project_path}")
    artifact_collection = api.artifact_collection(type_name=artifact_type, name=artifact_name_project_path)

    logger.info(f"Searching for best model in {artifact_name_project_path} using metric: {metric_name}")
    # Find the artifact model with the best metric
    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        logger.info(f"  - Checking artifact: {artifact.name} with {metric_name}={artifact.metadata.get(metric_name)}")
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logging.error("No model found in registry.")
        return

    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")

    # Link the best model in the project team to the organization registry collection with the aliases 'best' and 'staging'
    best_artifact.link(
        target_path=artifact_registry_path,
        aliases=["best", "staging"],
    )
    best_artifact.save()
    logger.info(f"Artifact {best_artifact.name} linked to {artifact_registry_path} with aliases 'best' and 'staging'.")


if __name__ == "__main__":
    typer.run(stage_best_model_to_registry)
