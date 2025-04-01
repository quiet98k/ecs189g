""" You can define utilities here to be shared across different parts.
"""

import torchvision
from helpers import Model
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks

rotate = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=(1, 2))), 
    transforms.Lambda(lambda x: x.contiguous().reshape(-1))
])