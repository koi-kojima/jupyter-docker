#!/usr/bin/env python3
import os
import sys
import time

# Graph
import matplotlib.pyplot as plt
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
# Utility
from tqdm import trange, tqdm

N = 40  # Train iteration count


def main(device: torch.device, mini_batch: int, is_gpu: bool) -> None:
    workers = 4 if is_gpu else 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = torchvision.datasets.QMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=mini_batch, shuffle=True, num_workers=workers, pin_memory=True
    )
    test = torchvision.datasets.QMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=mini_batch, shuffle=False, num_workers=workers, pin_memory=True
    )
    train_and_evaluate(device, is_gpu, train_loader, test_loader)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))  # 28x28x1 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, (3, 3))  # 26x26x32 -> 24x24x64
        self.pool1 = nn.MaxPool2d(2, 2)  # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 256)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x.view(-1, 12 * 12 * 64)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def evaluate(device, net, is_gpu, loss_func, data_loader, get_loss=False):
    total, correct, loss = 0, 0, 0
    p = tqdm(data_loader) if not get_loss else data_loader
    for inputs, labels in p:
        inputs = inputs.to(device, non_blocking=is_gpu)
        labels = labels.to(device, non_blocking=is_gpu)
        outputs = net(inputs)
        _, prediction = torch.max(outputs, 1)
        total += outputs.size(0)
        correct += (labels == prediction).sum().item()
        if get_loss:
            loss += loss_func(outputs, labels).item()
    if get_loss:
        return total, correct, loss
    else:
        return total, correct


def _print(content, title: str):
    print("*" * 10 + " " + title + " " + "*" * 10)
    print(content)
    print("*" * (20 + len(title) + 2))


def train_and_evaluate(device: torch.device, is_gpu: bool,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader):
    net = Net().to(device, )
    _print(net, "Network")
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    loss_log, accuracy_log = [], []
    test_epochs, test_accuracy_log, test_loss_log = [], [], []
    started = time.time()
    for epoch in trange(N):
        rl = 0.0
        _total, _correct = 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss: torch.Tensor = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _, prediction = torch.max(outputs, 1)
                _total += outputs.size(0)
                _correct += (labels == prediction).sum().item()

            rl += loss.item()
        loss_log.append(rl)
        accuracy_log.append(_correct / _total)
        if epoch % 20 == 19 or epoch < 5:
            with torch.no_grad():
                test_total, test_correct, test_loss = evaluate(device, net, is_gpu, loss_func, test_loader, True)
                test_accuracy_log.append(test_correct / test_total)
                test_loss_log.append(test_loss)
                test_epochs.append(epoch)
            tqdm.write(f"[{epoch=}] loss={rl:.3f} test={test_loss:.3f} "
                       f"correct={_correct} total={_total}")
    took = time.time() - started
    _print(f"Learning {N} steps took {took}, average={took / N}.", "Result")

    make_graph(N, accuracy_log, device, loss_log, test_accuracy_log, test_epochs, test_loss_log)
    started = time.time()
    with torch.no_grad():
        train_total, train_correct = evaluate(device, net, is_gpu, loss_func, train_loader)
        test_total, test_correct = evaluate(device, net, is_gpu, loss_func, test_loader)
        _print(
            f"Evaluation took {time.time() - started}.\n"
            f"Train Accuracy {train_correct / train_total:.3f} ({train_correct=}, {train_total=})\n"
            f"Test Accuracy {test_correct / test_total:.3f} ({test_correct=}, {test_total=})",
            "Evaluation"
        )


def make_graph(epochs, accuracy_log, device, loss_log, test_accuracy_log, test_epochs, test_loss_log):
    fig: plt.Figure = plt.figure(figsize=(8, 6))
    ax1: plt.Axes = fig.add_subplot(1, 1, 1)
    ax2: plt.Axes = ax1.twinx()
    ax1.plot(range(epochs), loss_log, label="TRAIN Loss")
    ax1.plot(test_epochs, test_loss_log, label="TEST Loss", ls="dashed")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(range(epochs), accuracy_log, label="TRAIN Accuracy", color="green")
    ax2.plot(test_epochs, test_accuracy_log, label="TEST Accuracy",
             color="purple", ls="dashed")
    ax2.set_ylim(min(min(accuracy_log), min(test_accuracy_log)) * 0.95, 1)
    ax2.set_ylabel("Accuracy")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='right')
    ax1.grid()
    ax1.set_title(f"Learning Loss and Accuracy of QMNIST on {device}")
    fig.savefig("QMNIST_train.pdf", dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Usage - ./mnist.py [cpu|gpu]")
        exit(1)
    _device_name = sys.argv[1]
    _batch = 128
    _is_gpu = False
    if _device_name.lower() == "gpu":
        if torch.cuda.is_available():
            print("GPU is supported and selected.")
            _d = torch.device("cuda:0")
            _batch = 512
            _is_gpu = True
        else:
            print("GPU is NOT supported. Using CPU.")
            _d = torch.device("cpu")
    elif _device_name.lower() == "cpu":
        if torch.cuda.is_available():
            print("GPU is supported but CPU is manually selected.")
        else:
            print("GPU is NOT supported. CPU is selected.")
        _d = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(os.cpu_count())
        _print(torch.__config__.parallel_info(), "Parallel Info")
    else:
        raise ValueError(f"Unknown device, {_device_name}.")
    if len(sys.argv) > 2:
        _batch = int(sys.argv[2])
    print(f"Device={_d}, Batch Size={_batch}")
    main(_d, _batch, _is_gpu)
