import typer
import wandb
from typing import List, Optional
from text_detect.wandb_functions import load_wandb_env_vars, get_artifact_project_path, get_registry_collection_path

from dotenv import load_dotenv

load_dotenv()


def link_model(
    artifact: str = typer.Argument(..., help="Artifact path in format 'name:version'"),
    aliases: Optional[List[str]] = typer.Option(
        None, "--alias", "-a", help="Aliases to apply to the model. Can be specified multiple times."
    ),
    full_registry_path: Optional[bool] = typer.Option(
        False, "--full-registry-path", "-f", help="Whether to use the full registry path instead of the default."
    ),
) -> None:
    """
    Link a specific model to the model registry with the given aliases.

    Both the team project and the organization registry and collection must be specified in the environment variables.

    Args:
        artifact: Artifact path in format 'name:version'
        aliases: List of aliases to apply to the model, for example 'staging'
    """
    if not artifact:
        typer.echo("No artifact path provided. Exiting.")
        return

    # Load environment variables
    api_key, team_name, project_name, entity_name, registry_name, collection_name = load_wandb_env_vars()

    # Initialize W&B API
    api = wandb.Api(api_key=api_key)

    if full_registry_path:
        # Load artifact from full registry path and link to organization registry with given aliases
        artifact_registry_path = artifact
        artifact = api.artifact(artifact_registry_path)
        artifact.link(target_path=artifact_registry_path, aliases=aliases)
        artifact.save()
        typer.echo(f"Artifact {artifact_registry_path} linked with aliases {aliases}.")
    else:
        # Extract artifact name and version
        artifact_name, artifact_name_version = artifact.split(":")

        # Define artifact paths for linking between team project and organization registry
        artifact_project_path = get_artifact_project_path(team_name, project_name, artifact_name, artifact_name_version)
        artifact_registry_path = get_registry_collection_path(entity_name, registry_name, collection_name)

        # Load artifact from team project and link to organization registry with given aliases
        artifact = api.artifact(artifact_project_path)
        artifact.link(target_path=artifact_registry_path, aliases=aliases)
        artifact.save()
        typer.echo(f"Artifact {artifact_project_path} linked to {artifact_registry_path} with aliases {aliases}.")


if __name__ == "__main__":
    typer.run(link_model)
