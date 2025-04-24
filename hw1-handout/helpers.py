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

def load_testset() -> torchvision.datasets.MNIST:
    return load_dataset(train=False)

def load_edit_dataset(
    root: str='./data/mnist_c/fog/',
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    shape: Tuple[int, ...] = (784,),
    size: Optional[int] = None,
) -> torch.utils.data.TensorDataset:
    root = Path(root)
    images = torch.from_numpy(np.load(root/'test_images.npy'))\
            .to(device=device, dtype=dtype)\
            .reshape(-1, *shape) / 255.
    labels = torch.from_numpy(np.load(root/'test_labels.npy'))\
            .to(device=device, dtype=torch.int64)

    if size is not None:
        images = images[:size]
        labels = labels[:size]

    return torch.utils.data.TensorDataset(
        images,
        labels,
    )

def load_edit_dataset_part_1() -> torch.utils.data.TensorDataset:
    dataset = load_edit_dataset()
    dataset.tensors = (
        dataset.tensors[0][:4000:200],
        dataset.tensors[1][:4000:200],
    )
    return dataset


def load_edit_dataset_part_2() -> torch.utils.data.TensorDataset:
    return load_edit_dataset()

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

def save_model(
    model: nn.Sequential,
    path: str | Path
) -> None:
    torch.save(model.state_dict(), path)
    print(f"Saved model parameters to '{path}'.")


@torch.no_grad()
def test(model: nn.Module,
         testloader: Optional[torch.utils.data.DataLoader | torch.utils.data.Dataset] = None,
         *,
         device: Optional[torch.device]=None,
) -> Tuple[float, float]:

    if device is None:
        device = next(model.parameters()).device

    if testloader is None:
        testloader = load_testset()

    if isinstance(testloader, torch.utils.data.Dataset):
        testloader = torch.utils.data.DataLoader(
            testloader, batch_size=1000, shuffle=False
        )

    model.eval()
    correct = 0

    for data, target in tqdm(testloader, desc='testing', leave=False):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    total = len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    print(f'Accuracy: {correct}/{total} ({accuracy:.2%})\n')

    return accuracy
