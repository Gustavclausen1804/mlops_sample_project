from pathlib import Path

import torch
import typer

from mlops_sample_project.data import corrupt_mnist
from mlops_sample_project.model import FashionMinistClassifierModel
from mlops_sample_project.train import DEVICE


def evaluate(model_checkpoint: str = typer.Argument("model.pth")) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = FashionMinistClassifierModel().to(DEVICE)
    model_path = Path("./models") / model_checkpoint
    model.load_state_dict(torch.load(model_path))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
