import os

import torch
import typer

import wandb
from mnist.data import corrupt_mnist
from mnist.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix for E501: Move long string to a constant to keep function signature short
DEFAULT_REGISTRY = "s250269-danmarks-tekniske-universitet-dtu/wandb-registry-Mnist_models/corrupt_mnist_models:latest"


def evaluate(registry_path: str = DEFAULT_REGISTRY) -> None:
    """Evaluate a model directly from the W&B Model Registry."""
    print(f"Fetching model from registry: {registry_path}")

    # 1. Initialize W&B API and download the artifact
    api = wandb.Api()
    try:
        artifact = api.artifact(registry_path)
        # Download to a consistent local directory
        artifact_dir = artifact.download(root="downloaded_model")
        model_checkpoint = os.path.join(artifact_dir, "model.pth")
    except Exception as e:
        print(f"Error fetching artifact: {e}")
        return

    # 2. Load the model safely
    print(f"Loading checkpoint from: {model_checkpoint}")
    model = MyAwesomeModel().to(DEVICE)

    # Use weights_only=True for security best practices in newer Torch versions
    state_dict = torch.load(model_checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Data loading and evaluation loop
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    typer.run(evaluate)
