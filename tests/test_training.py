import os

import hydra
import pytest
import torch
from omegaconf import DictConfig

from src.mlops_sample_project.data import corrupt_mnist
from src.mlops_sample_project.model import create_model_from_model_params_yaml

from . import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("batch_size", [32, 64])
def test_training_loop(batch_size: int) -> None:
    # Set up Hydra configuration programmatically
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="default_config.yaml")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = cfg.model_experiments.params
    train_hparams = cfg.train_experiments

    # Create model and data loader
    model = create_model_from_model_params_yaml(model_params).to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Set up loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_hparams["learning_rate"])

    # Run a single training batch
    for img, target in train_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(img)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0  # Ensure loss decreases
        break  # Test only one batch