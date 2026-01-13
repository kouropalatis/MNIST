import os
from typing import List

import typer

import wandb


def link_model(artifact_path: str, aliases: List[str] = ["production"]) -> None:
    """
    Link a specific model artifact to the W&B model registry with a 'production' alias.

    Args:
        artifact_path: The full path to the artifact (entity/project/name:version).
        aliases: List of aliases to apply (default is ["production"]).
    """
    if not artifact_path:
        typer.echo("No artifact path provided. Exiting.")
        raise typer.Exit(code=1)

    # Initialize W&B API using environment variables
    # WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT must be set in your GitHub Secrets
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

    try:
        # Extract the artifact name from the provided path
        # Path format: "entity/project/artifact_name:version"
        _, _, artifact_name_version = artifact_path.split("/")
        artifact_name, _ = artifact_name_version.split(":")

        # Fetch the artifact from W&B
        artifact = api.artifact(artifact_path)

        # Define the target path in the Model Registry
        # Usually: "entity/model-registry/artifact_name"
        target_path = f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}"

        # Link the model and save the changes
        artifact.link(target_path=target_path, aliases=aliases)
        artifact.save()

        typer.echo(f"✅ Success: {artifact_path} linked to {target_path} as {aliases}")

    except Exception as e:
        typer.echo(f"❌ Error linking model: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(link_model)
