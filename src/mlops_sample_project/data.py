from pathlib import Path
import typer
import torch
import matplotlib.pyplot as plt  # only needed for plotting
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
from torch.utils.data import Dataset, TensorDataset


class CorruptMNISTDataset(Dataset):
    """Dataset for corrupt MNIST."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.train_images = []
        self.train_target = []
        self.test_images = None
        self.test_target = None


    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images."""
        return (images - images.mean()) / images.std()

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Preprocessing data...")

        # Load test data
        self.test_images = torch.load(self.data_path / "test_images.pt").unsqueeze(1).float()
        print(f"Shape of test_images: {self.test_images.shape}")
        self.test_target = torch.load(self.data_path / "test_target.pt").long()
        print(f"Shape of test_target: {self.test_target.shape}")

        # Load train data
        for i in range(6):
            img = torch.load(self.data_path / f"train_images_{i}.pt").unsqueeze(1).float()
            print(f"Shape of train_images_{i}: {img.shape}")
            tgt = torch.load(self.data_path / f"train_target_{i}.pt").long()
            print(f"Shape of train_target_{i}: {tgt.shape}")
            self.train_images.append(img)
            self.train_target.append(tgt)

        # Concatenate and save processed data
        self.train_images = torch.cat(self.train_images)
        self.train_target = torch.cat(self.train_target)
        
        
        # normalize the data
        self.train_images = self.normalize(self.train_images)
        self.test_images = self.normalize(self.test_images)
        
        train_set = TensorDataset(self.train_images, self.train_target)
        test_set = TensorDataset(self.test_images, self.test_target)

        torch.save(train_set, output_folder / "train_set.pt")
        torch.save(test_set, output_folder / "test_set.pt")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.train_images) if self.train_images else 0

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if not self.train_images:
            raise ValueError("Dataset is not loaded. Run preprocess first.")
        return self.train_images[index], self.train_target[index]


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    dataset = CorruptMNISTDataset(raw_data_path)
    dataset.preprocess(output_folder)


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


def main(raw_data_path: Path, output_folder: Path) -> None:
    preprocess(raw_data_path, output_folder)
    train_set: TensorDataset = torch.load(output_folder / "train_set.pt")
    test_set: TensorDataset = torch.load(output_folder / "test_set.pt")

    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")

    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])


if __name__ == "__main__":
    typer.run(main)
