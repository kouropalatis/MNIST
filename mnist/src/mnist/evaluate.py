import torch
import typer
from mnist.data import corrupt_mnist
from mnist.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Make sure your function takes model_checkpoint as an argument
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print(f"Evaluating model from: {model_checkpoint}")

    model = MyAwesomeModel().to(DEVICE)
    # Use map_location to ensure it loads on CPU if CUDA isn't available
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
    
    print(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    # 2. Use typer.run to properly handle the command line argument
    typer.run(evaluate)