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
    # 1. Environment Validation
    required_vars = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        typer.echo(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        raise typer.Exit(code=1)

    if not artifact_path:
        typer.echo("❌ Error: No artifact path provided. Exiting.")
        raise typer.Exit(code=1)

    # Initialize W&B API using environment variables
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

    try:
        # Extract the artifact name from the provided path
        # Path format: "entity/project/artifact_name:version"
        # Only split once on first colon to separate version
        if ":" not in artifact_path:
             raise ValueError(f"Artifact path '{artifact_path}' must contain a version (e.g., :v1 or :latest)")
             
        artifact_name_version = artifact_path.split("/")[-1]
        artifact_name = artifact_name_version.split(":")[0]

        # Fetch the artifact from W&B
        typer.echo(f"Using artifact path: {artifact_path}")
        artifact = api.artifact(artifact_path)

        # Define the target path in the Model Registry
        target_entity = os.getenv('WANDB_ENTITY')
        target_path = f"{target_entity}/model-registry/{artifact_name}"
        typer.echo(f"Linking to registry path: {target_path}")

        # Link the model and save the changes
        artifact.link(target_path=target_path, aliases=aliases)
        artifact.save()

        typer.echo(f"✅ Success: {artifact_path} linked to {target_path} as {aliases}")

    except Exception as e:
        typer.echo(f"❌ Error linking model: {e}")
        # Print full exception for debugging in CI logs
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(link_model)
