import os
import matplotlib.pyplot as plt
import torch
import typer
from mnist.model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model embeddings using t-SNE."""
    print(f"Loading model from {model_checkpoint}...")
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    # Model Surgery: replace the final fully connected layer with Identity
    # to get the 128-dimensional features instead of 10-class probabilities
    model.fc1 = torch.nn.Identity() 

    print("Loading processed test data...")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    print("Extracting features...")
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            images = images.to(DEVICE)
            predictions = model(images)
            embeddings.append(predictions.cpu())
            targets.append(target)
            
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    print("Running dimensionality reduction (t-SNE)...")
    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i), alpha=0.6)
    
    plt.legend()
    plt.title("t-SNE Visualization of MNIST Embeddings")
    
    # Ensure folder exists and save
    os.makedirs("reports/figures", exist_ok=True)
    save_path = f"reports/figures/{figure_name}"
    plt.savefig(save_path)
    print(f"Success! Figure saved to {save_path}")

if __name__ == "__main__":
    typer.run(visualize)