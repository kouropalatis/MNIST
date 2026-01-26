import os
import torch
import typer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
from mnist.data import corrupt_mnist
from mnist.model import MyAwesomeModel
from omegaconf import OmegaConf
from hydra import compose, initialize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    cfg,
    save_path: Optional[str] = None,
) -> None:
    """Train model using Hydra config."""
    # Print config to verifying loading
    print(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Extract params from config
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    seed = cfg.training.get("seed", 42)
    
    torch.manual_seed(seed)

    run = wandb.init(
        project="corrupt_mnist",
        config=dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        preds_list, targets_list = [], []

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            acc = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train/loss": loss.item(), "train/accuracy": acc})

            preds_list.append(y_pred.detach().cpu())
            targets_list.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")

        epoch_preds = torch.cat(preds_list, 0)
        epoch_targets = torch.cat(targets_list, 0)
        preds_labels = epoch_preds.argmax(dim=1)

        metrics = {
            "accuracy": accuracy_score(epoch_targets, preds_labels),
            "precision": precision_score(epoch_targets, preds_labels, average="weighted"),
            "recall": recall_score(epoch_targets, preds_labels, average="weighted"),
            "f1": f1_score(epoch_targets, preds_labels, average="weighted"),
        }
        wandb.log(metrics)

    os.makedirs("models", exist_ok=True)
    actual_save_path = save_path if save_path else "models/model.pth"
    torch.save(model.state_dict(), actual_save_path)

    artifact = wandb.Artifact(name="corrupt_mnist_model", type="model", metadata=metrics)
    artifact.add_file(actual_save_path)
    run.log_artifact(artifact)
    run.link_artifact(
        artifact=artifact, target_path="wandb-registry-Mnist_models/corrupt_mnist_models", aliases=["latest"]
    )
    run.finish()

app = typer.Typer()

from typing import Optional

@app.command()
def main(
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    # Load config from configs/config.yaml using Hydra
    # Pointing to "../../configs" assuming running from project root or src/mnist
    # We use path relative to this file's location or project root.
    # The safest is to rely on standard Hydra usage or explicit path.
    # Ref uses: with initialize(version_base=None, config_path="../../configs"):
    
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    # Override config with CLI args if provided
    if epochs:
        cfg.training.epochs = epochs
    if lr:
        cfg.training.lr = lr
    if batch_size:
        cfg.training.batch_size = batch_size

    train(cfg)

if __name__ == "__main__":
    app()
