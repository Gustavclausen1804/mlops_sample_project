import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import typer
from omegaconf import OmegaConf

from mlops_sample_project.data import corrupt_mnist
from mlops_sample_project.model import FashionMinistClassifierModel, create_model_from_model_params_yaml

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")
project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
config_path = str(project_root / "configs")

@hydra.main(config_path=config_path, config_name="default_config.yaml")
def train(config) -> None:
    """Train a model on MNIST."""
    
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.train_experiments
    model_params_yaml = config.model_experiments.params
    
    torch.manual_seed(hparams["seed"])
    print(f"Current working directory: {os.getcwd()}")

    
    epochs : int = hparams["epochs"]
    lr : float = hparams["learning_rate"]
    batch_size : int = hparams["batch_size"]    

    model = create_model_from_model_params_yaml(model_params_yaml).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

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
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("reports/figures"):
        os.makedirs("reports/figures")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()

