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

def test_performance(model):
    minst_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=rotate)
    data_loader = torch.utils.data.DataLoader(minst_dataset, batch_size=64, shuffle=True)
    
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    accuracy = 100 * correct_test / total_test

    return accuracy