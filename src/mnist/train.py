import matplotlib.pyplot as plt
import torch
import typer
import wandb
import os
from mnist.data import corrupt_mnist
from mnist.model import MyAwesomeModel
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST and log it as a W&B Artifact."""
    print(f"Training on {DEVICE}")
    
    # 1. Start the run
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        preds, targets = [], []
        
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            # Simple scalar logging
            acc = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train/loss": loss.item(), "train/accuracy": acc})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
                # Log sample images and gradients
                wandb.log({
                    "media/images": wandb.Image(img[:5].detach().cpu(), caption="Input images"),
                    "media/gradients": wandb.Histogram(
                        torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu()
                    )
                })

        # 2. Log ROC Curves at end of epoch
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        
        # (ROC Curve logic removed for brevity but keep it in your local file)
        # wandb.log({"media/roc": wandb.Image(plt)})

    # 3. Calculate Final Performance Metrics
    preds_labels = preds.argmax(dim=1)
    metrics = {
        "accuracy": accuracy_score(targets, preds_labels),
        "precision": precision_score(targets, preds_labels, average="weighted"),
        "recall": recall_score(targets, preds_labels, average="weighted"),
        "f1": f1_score(targets, preds_labels, average="weighted"),
    }

    # 4. Save and Log Model Artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pth"
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="CNN trained on corrupt MNIST",
        metadata=metrics  # Attach metrics directly to the artifact!
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    
    print(f"Model logged to W&B as an artifact with metrics: {metrics}")
    run.finish()

if __name__ == "__main__":
    typer.run(train)