import os
import shlex

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "text_detect"
PYTHON_VERSION = "3.11"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)
    ctx.run('pip install -e .["types"]', echo=True, pty=not WINDOWS)


@task
def pre_commit_install(ctx: Context) -> None:
    """Install pre-commit hooks."""
    ctx.run("pre-commit install", echo=True, pty=not WINDOWS)


@task
def download_data(ctx: Context) -> None:
    """Download data from Kaggle."""
    ctx.run("sh data/downloadKaggleDataset.sh", echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def analyze_data(ctx: Context) -> None:
    """Analyze data."""
    ctx.run(f"python src/{PROJECT_NAME}/analysis.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate_model(ctx: Context, artifact="", config="default"):
    """
    Evaluate a model using the specified artifact and config.

    Args:
        artifact: Artifact path in format 'name:version'
        config: Name of config file to use (without .yaml extension)
    """
    if not artifact:
        print("Error: Please provide an artifact (e.g. 'llm-detector-model:v4')")
        return

    command = f"python src/{PROJECT_NAME}/evaluate.py {shlex.quote(artifact)} --config {shlex.quote(config)}"

    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def link_to_registry(ctx: Context, artifact="", aliases=None):
    """
    Link a specific team project model to the organization registry collection with the given aliases.

    Args:
        artifact: Artifact path in format 'name:version'
        aliases: Optional comma-separated list of aliases (no spaces)
    """
    if not artifact:
        print("Error: Please provide an artifact (e.g. 'llm-detector-model:v4')")
        return

    # Build command
    command = f"python src/{PROJECT_NAME}/link_model.py {shlex.quote(artifact)}"

    # Add aliases if provided
    if aliases:
        alias_args = " ".join(f"-a {alias}" for alias in aliases.split(","))
        command = f"{command} {alias_args}"

    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def stage_best_model(ctx: Context, artifact_name="", type="model", metric="best_val_accuracy", higher_is_better=True):
    """
    Stage the best model to the registry based on the specified metric.

    Args:
        artifact_name: Name of the model to be registered, e.g. 'llm-detector-model'
        artifact_type: Type of the artifact to be registered.
        metric: Metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.
    """
    if not artifact_name:
        print("Error: Please provide an artifact name")
        return

    higher_is_better_flag = "--higher-is-better" if higher_is_better else ""
    lower_is_better_flag = "" if higher_is_better else "--lower-is-better"

    command = (
        f"python src/{PROJECT_NAME}/stage_best_model.py "
        f"{artifact_name} "
        f"--type {type} "
        f"--metric {metric} "
        f"{higher_is_better_flag} {lower_is_better_flag}"
    ).strip()

    ctx.run(command, echo=True, pty=not WINDOWS)


# Testing commands
@task
def unittests(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run --source=src/text_detect -m pytest src/tests/unittests", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


# Docker commands
@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
