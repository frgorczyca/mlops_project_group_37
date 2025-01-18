import os
import wandb
from dotenv import load_dotenv

load_dotenv()

# Set the artifact_name and version
artifact_name = "llm-detector-model"
version = "latest"

api_key = os.getenv("WANDB_API_KEY")
entity_name = os.getenv("WANDB_ENTITY")
team_name = os.getenv("WANDB_TEAM")
project_name = os.getenv("WANDB_PROJECT")
registry_name = "Text detect registry"
collection_name = "Trained LLM detector"

artifact_project_path = f"{team_name}/{project_name}/{artifact_name}:{version}"
artifact_registry_path = f"{entity_name}/wandb-registry-{registry_name}/{collection_name}"

api = wandb.Api(
    api_key=os.getenv("WANDB_API_KEY")
)
artifact = api.artifact(artifact_project_path)
artifact.link(target_path=artifact_registry_path)
artifact.save()