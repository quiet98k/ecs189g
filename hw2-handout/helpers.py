""" Helper definitions for homework 1.
    Please do not modify this file.
"""
from typing import Literal, Optional, Tuple
from pathlib import Path
from typing import Optional, Self
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda t: t.flatten(-3,-1)),
])

def load_dataset(root: str='./data', train: bool=True) -> torchvision.datasets.MNIST:
    """ Load the MNIST dataset with the specified transformations. """
    return torchvision.datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=dataset_transform
    )

def load_model(
    path: Optional[str | Path] = './model.pt',
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    ).to(device=device, dtype=dtype)
    if path is not None:
        model.load_state_dict(
            torch.load(path, weights_only=True, map_location=device)
        )
    return model

class AdversarialExamples:
    images: torch.Tensor
    adv_images: torch.Tensor
    targets: torch.Tensor

    def __init__(self,
                 images: torch.Tensor,
                 adv_images: torch.Tensor,
                 targets: torch.Tensor,
    ) -> None:
        N, C = images.shape

        assert C == 784, \
            f'Expected @images to have shape (N, 784), got {images.shape}'

        assert targets.shape == (N,), \
            f'Expected @targets to have shape (N,), got {targets.shape}'

        assert targets.dtype == torch.int64, \
            f'Expected @targets to have dtype torch.int64, got {targets.dtype}'

        assert adv_images.shape == images.shape, \
            f'Expected @adv_images to have shape {images.shape}, got {adv_images.shape}'

        self.images = images
        self.targets = targets
        self.adv_images = adv_images

    def __len__(self) -> int:
        return self.targets.shape[0]

    def save(self, path: str | Path) -> None:
        """ Save the adversarial examples to a file.

        Args:
            path (str | Path): file path to save the adversarial examples.
        """
        torch.save(vars(self), path)
        print(f"Saved {len(self)} adversarial examples to '{path}'.")

    @classmethod
    def load(cls, path: str | Path) -> 'AdversarialExamples':
        """ Load adversarial examples from a file.

        Args:
            path (str | Path): file path to load the adversarial examples.

        Returns:
            AdversarialExample: loaded adversarial examples.
        """
        adv = cls(**torch.load(path, weights_only=True, map_location='cpu'))
        # print(f"Loaded {len(adv)} adversarial examples from '{path}'.")
        return adv

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Self:
        self.images = self.images.to(device=device, dtype=dtype)
        self.adv_images = self.adv_images.to(device=device, dtype=dtype)
        self.targets = self.targets.to(device=device)
        return self
