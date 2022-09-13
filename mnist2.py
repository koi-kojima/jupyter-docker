#!/usr/bin/env python3
import argparse

import mnist_dataset
import pytorch_lightning as pl
import torch.optim
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import *
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        activation = nn.PReLU  # Change here to `nn.ReLU` to use ReLU.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # 28x28x1 -> 26x26x32
            activation(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),  # 26x26x32 -> 24x24x64
            activation(),
            nn.MaxPool2d(2, 2),  # 24x24x64 -> 12x12x64
        )
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 256),
            activation()
        )
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 10),
            activation(),
        )
        self.soft_max = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

    def feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def _do_prediction(self, batch):
        x, y = batch
        y_prediction = self.soft_max(self.forward(x))
        loss = self.loss_function(y_prediction, y)
        self.accuracy(y_prediction, y)
        return loss

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        loss = self._do_prediction(batch)
        self.log("train_accuracy", self.accuracy)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        loss = self._do_prediction(batch)
        self.log("val_accuracy", self.accuracy)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        loss = self._do_prediction(batch)
        self.log("test_accuracy", self.accuracy)
        self.log("test_loss", loss)
        return loss


class MNISTLoader(pl.LightningDataModule):
    _train_data: Dataset
    _validation_data: Dataset
    _test_data: Dataset

    def __init__(self, mini_batch: int, dataset_name: str):
        super().__init__()
        self.mini_batch = mini_batch
        self.dataset_name = dataset_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.validation_ratio = 0.2

    def prepare_data(self, stage: Optional[str] = None) -> None:
        dataset_class = mnist_dataset.get_mnist_dataset(self.dataset_name)
        train_all = dataset_class(root="./data", train=True, download=True, transform=self.transform)
        train_size = len(train_all)
        self._train_data, self._validation_data = torch.utils.data.random_split(train_all, [
            int(train_size * (1 - self.validation_ratio)), int(train_size * self.validation_ratio)])
        self._test_data = dataset_class(root="./data", train=False, download=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_data, batch_size=self.mini_batch, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._validation_data, batch_size=self.mini_batch, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_data, batch_size=self.mini_batch, num_workers=4)


def main():
    parser = argparse.ArgumentParser(description="Run MNIST.")
    parser.add_argument("-b", "--batch", type=int, help="batch size (default=512)", default=512)
    parser.add_argument("-e", "--epoch", type=int, help="epoch count", default=5)
    parser.add_argument("-d", "--dataset", type=str, help="dataset name (default=MNIST)", default="MNIST")
    parser.add_argument("--device", type=str, help="device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    print(f"Batch Size={args.batch}, Dataset={args.dataset}, Epoch={args.epoch}")

    model = MNISTModel()
    loader = MNISTLoader(args.batch, args.dataset)
    trainer = Trainer(max_epochs=args.epoch, accelerator=args.device, devices=1)
    trainer.fit(model, loader)
    trainer.validate(model, loader)
    trainer.test(model, loader)


if __name__ == '__main__':
    main()
