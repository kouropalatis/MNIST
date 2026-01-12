import os

import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mnist.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model embeddings using t-SNE."""
    print(f"Loading model from {model_checkpoint}...")
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    # Model Surgery: Added type ignore for assignment
    model.fc1 = torch.nn.Identity()  # type: ignore[assignment]

    print("Loading processed test data...")
    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_target = torch.load("data/processed/test_target.pt", weights_only=True)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings_list, targets_list = [], []
    print("Extracting features...")
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            images = images.to(DEVICE)
            predictions = model(images)
            embeddings_list.append(predictions.cpu())
            targets_list.append(target)

        # Unique names for the numpy conversion to satisfy Mypy
        embeddings_np = torch.cat(embeddings_list).numpy()
        targets_np = torch.cat(targets_list).numpy()

    print("Running dimensionality reduction (t-SNE)...")
    if embeddings_np.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings_np = pca.fit_transform(embeddings_np)

    tsne = TSNE(n_components=2)
    embeddings_final = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets_np == i
        plt.scatter(embeddings_final[mask, 0], embeddings_final[mask, 1], label=str(i), alpha=0.6)

    plt.legend()
    plt.title("t-SNE Visualization of MNIST Embeddings")
    os.makedirs("reports/figures", exist_ok=True)
    save_path = f"reports/figures/{figure_name}"
    plt.savefig(save_path)
    print(f"Success! Figure saved to {save_path}")


if __name__ == "__main__":
    typer.run(visualize)
