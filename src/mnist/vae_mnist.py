import logging
import os

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import wandb

log = logging.getLogger(__name__)


def loss_function(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


@hydra.main(config_path="../../config", config_name="config.yaml", version_base="1.3")
def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.setup.seed)
    # Using lowercase to avoid N806 (non-constant in function)
    device = torch.device(cfg.setup.device if torch.cuda.is_available() else "cpu")

    # Correct Hydra to dict conversion for WandB
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="mnist_vae", config=wandb_config)  # type: ignore[arg-type]

    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="~/datasets", transform=mnist_transform, train=True, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    log.info(f"Start training VAE on {device}...")
    model.train()
    for epoch in range(cfg.training.epochs):
        overall_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

            if batch_idx % 100 == 0:
                log.info(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item() / len(x):.4f}")

        # Explicitly cast to Sized if mypy complains about len()
        avg_loss = overall_loss / len(train_loader.dataset)  # type: ignore[operator, arg-type]
        log.info(f"Epoch {epoch + 1} complete! Average Loss: {avg_loss:.4f}")
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch + 1})

    output_dir = HydraConfig.get().runtime.output_dir
    torch.save(model, os.path.join(output_dir, "trained_model.pt"))
    wandb.finish()


if __name__ == "__main__":
    train()
