from typing import Type

from torchvision.datasets import MNIST, QMNIST, FashionMNIST, KMNIST

__all__ = [
    "get_mnist_dataset",
]


# noinspection SpellCheckingInspection
def get_mnist_dataset(dataset_name: str) -> Type[MNIST]:
    dataset_name = dataset_name.casefold()
    if not dataset_name or dataset_name == "QMNIST".casefold() or dataset_name == "MNIST".casefold():
        return QMNIST
    elif dataset_name == "FashionMNIST".casefold():
        return FashionMNIST
    elif dataset_name == "KMNIST".casefold():
        return KMNIST
    else:
        raise ValueError(f"Unknown dataset {dataset_name=}")
