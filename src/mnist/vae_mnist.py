import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import hydra
import wandb
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

log = logging.getLogger(__name__)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld

@hydra.main(config_path="../../config", config_name="config.yaml", version_base="1.3")
def train(cfg: DictConfig):
    # 1. Reproducibility
    torch.manual_seed(cfg.setup.seed)
    DEVICE = torch.device(cfg.setup.device if torch.cuda.is_available() else "cpu")

    # 2. WandB
    wandb.init(project="mnist_vae", config=OmegaConf.to_container(cfg, resolve=True))

    # 3. Data Loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="~/datasets", transform=mnist_transform, train=True, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # 4. Instantiate Model
    # Hydra builds the Model and its sub-components (Encoder/Decoder) automatically
    model = instantiate(cfg.model).to(DEVICE)

    # 5. Instantiate Optimizer
    # We pass params=model.parameters() as an extra argument
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    log.info(f"Start training VAE on {DEVICE}...") 
    model.train()
    for epoch in range(cfg.training.epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(DEVICE)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

            if batch_idx % 100 == 0:
                log.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item() / len(x):.4f}")

        avg_loss = overall_loss / (len(train_loader.dataset))
        log.info(f"Epoch {epoch + 1} complete! Average Loss: {avg_loss:.4f}") 
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch + 1})

    # 6. Logging Visuals & Saving
    # (Existing logging and saving logic remains here)
    output_dir = HydraConfig.get().runtime.output_dir
    torch.save(model, os.path.join(output_dir, "trained_model.pt"))
    wandb.finish()

if __name__ == "__main__":
    train()