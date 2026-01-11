import torch
import typer
import matplotlib.pyplot as plt
from mnist.model import MyAwesomeModel
from mnist.data import corrupt_mnist  # Import from your new modular data file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    print(f"Training on {DEVICE}")
    model = MyAwesomeModel().to(DEVICE)
    
    # This now loads the processed, normalized tensors
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            # Record stats
            statistics["train_loss"].append(loss.item())
            acc = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(acc)

    # Save to the Cookiecutter 'models' folder
    torch.save(model.state_dict(), "models/model.pth")
    print("Training complete. Model saved to models/model.pth")

if __name__ == "__main__":
    app()