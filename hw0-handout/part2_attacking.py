""" Implement this script so that
    ```
    uv run part2_attacking.py
    ```
    will find an adversarial example for your trained DNN from Part 1.

    You could use the following code to load your trained model:
    ```
    from helpers import Model
    model = Model().load('artifacts/model.pt')
    ```

    You could use the following code to save the adversarial example:
    ```
    from helpers import AdversarialExamples
    AdversarialExamples(images=images,
                        adv_images=adv_images,
                        targets=targets,
    ).save('artifacts/adversarial_examples.pt')
    ```

    See README.md for the requirements.
"""
import torchvision
from helpers import Model
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks
from utils import rotate

model = Model().load('artifacts/model.pt')

minst_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=rotate)
train_loader = torch.utils.data.DataLoader(minst_train_dataset, batch_size=1, shuffle=True)

image, label = next(iter(train_loader))

attack = torchattacks.PGD(model, eps=1e-2, alpha=1e-3, steps=40)
adv_img = attack(image, label)

perturbation = adv_img - image
linf_norm = torch.norm(perturbation, p=float('inf'))
print(f"L∞ Norm of perturbation: {linf_norm.item()}")

# Get predictions for the original and adversarial images
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    original_logits = model(image)
    adv_logits = model(adv_img)

    # Get the predicted classes
    original_pred = torch.argmax(original_logits, dim=1).item()
    adv_pred = torch.argmax(adv_logits, dim=1).item()

print(f"Original Label: {label.item()}, Original Prediction: {original_pred}")
print(f"Adversarial Prediction: {adv_pred}")

from helpers import AdversarialExamples
AdversarialExamples(images=image,
                    adv_images=adv_img,
                    targets=label,
).save('artifacts/adversarial_examples.pt')