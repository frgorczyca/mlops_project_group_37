import os
import shutil

from text_detect.model import LLMDetector


def load_wandb_env_vars() -> tuple:
    """Load environment variables for W&B API."""
    api_key = os.getenv("WANDB_API_KEY")
    entity_name = os.getenv("WANDB_ENTITY")
    team_name = os.getenv("WANDB_TEAM")
    project_name = os.getenv("WANDB_PROJECT")
    registry_name = os.getenv("WANDB_REGISTRY")
    collection_name = os.getenv("WANDB_COLLECTION")

    return api_key, team_name, project_name, entity_name, registry_name, collection_name


def get_artifact_project_path(team_name, project_name, artifact_name, artifact_name_version):
    """Get the team project artifact path, for example 'my-team/my-project/my-artifact:latest'."""
    return f"{team_name}/{project_name}/{artifact_name}:{artifact_name_version}"


def get_artifact_name_project_path(team_name, project_name, artifact_name):
    """Get the team project artifact name path (doesn't include version), for example 'my-team/my-project/my-artifact'."""
    return f"{team_name}/{project_name}/{artifact_name}"


def get_registry_collection_path(entity_name, registry_name, collection_name):
    """Get the organization registry collection path, for example 'my-organization/wandb-registry-my-registry>/my-collection'."""
    return f"{entity_name}/wandb-registry-{registry_name}/{collection_name}"


def load_download_artifact_model(cfg, api, artifact_project_path):
    """Load and download the artifact and model from W&B project artifact."""
    artifact = api.artifact(artifact_project_path)
    artifact.download(root="models/downloads")
    file_name = artifact.files()[0].name
    model_path = os.path.join("models/downloads", file_name)
    print(f"Downloaded model to {model_path}")
    return artifact, LLMDetector.load_from_checkpoint(model_path, cfg=cfg)


def cleanup_downloaded_model():
    """Remove only the contents within the downloads directory, keeping the directory itself."""
    download_path = "models/downloads"
    if os.path.exists(download_path):
        for item in os.listdir(download_path):
            item_path = os.path.join(download_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Removed {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed {item_path}")
