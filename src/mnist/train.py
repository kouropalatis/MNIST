import os

import torch
import typer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
from mnist.data import corrupt_mnist
from mnist.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    run = wandb.init(project="corrupt_mnist")

    # Casting to satisfy potential type mismatches from config
    lr = float(wandb.config.lr)
    batch_size = int(wandb.config["batch-size"])
    epochs = int(wandb.config.epochs)

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
                images = [wandb.Image(im.cpu()) for im in img[:5]]

                # Convert gradient tensor to numpy for Histogram
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                wandb.log({"media/images": images, "media/gradients": wandb.Histogram(grads.cpu().numpy().tolist()),})

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
    model_path = "models/model.pth"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(name="corrupt_mnist_model", type="model", metadata=metrics)
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    run.link_artifact(
        artifact=artifact, target_path="wandb-registry-Mnist_models/corrupt_mnist_models", aliases=["latest"]
    )
    run.finish()


if __name__ == "__main__":
    typer.run(train)
