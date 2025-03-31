""" Helper definitions for homework 0.
    Please do not modify this file.
"""

from pathlib import Path
from typing import Optional, Self
import torch
from torch import nn
import torchvision

class Model(nn.Module):
    """ DNN model for MNIST-related parts in homework 0.
        Please do not modify this class.
    """

    layers: nn.Sequential

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward function for @Model.

        Args:
            x (torch.Tensor): Input images in shape (*N, 784).

        Returns:
            torch.Tensor: Logits in shape (*N, 10).
        """
        return self.layers(x)

    def save(self, path: str | Path) -> None:
        """ Save the model parameters to a file.

        Args:
            path (str | Path): file path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"Saved model parameters to '{path}'.")

    def save_layer(self, layer: int, path: str | Path) -> None:
        """ Save the model layer @self.layers[@layer] parameters to a file.

        Args:
            layer (int): layer index. E.g., -1 for the last layer.
            path (str | Path): file path to save the model.
        """
        torch.save(self.layers[layer].state_dict(), path)
        print(f"Saved layer {layer} parameters to '{path}'.")

    def load(self, path: str | Path) -> 'Model':
        """ Load the model parameters from a file.

        Args:
            path (str | Path): file path to load the model.

        Returns:
            Model: @self with parameters loaded.
        """
        self.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
        # print(f"Loaded model parameters from '{path}'.")
        return self

    def load_layer(self, layer: int, path: str | Path) -> 'Model':
        """ Load the model layer @self.layers[@layer] parameters from a file.

        Args:
            layer (int): layer index. E.g., -1 for the last layer.
            path (str | Path): file path to load the model.

        Returns:
            Model: @self with parameters loaded.
        """
        self.layers[layer].load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
        # print(f"Loaded layer {layer} parameters from '{path}'.")
        return self


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


@torch.no_grad()
def save_img(tensor: torch.Tensor, path: str | Path, shape=(1,28,28)) -> None:
    torchvision.utils.save_image(tensor.reshape(*shape), path)
