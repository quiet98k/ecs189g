""" Implement this script so that
    ```
    uv run part1_training.py
    ```
    will train a DNN on a *rotated* MNIST dataset.

    You could use the following code to create a model:
    ```
    from helpers import Model
    model = Model()
    ```
    and use the following code to save the trained model parameters:
    ```
    model.save('artifacts/model.pt')
    ```

    See README.md for the requirements.
"""

from helpers import Model
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



model = Model()

rotate = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=-1, dims=(1, 2))), 
    transforms.Lambda(lambda x: x.contiguous().reshape(-1))
])

# # Load the MNIST dataset with and without the rotation
# mnist_original = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# mnist_rotated = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=rotate)

# # Visualize a few samples
# def visualize_samples(dataset1, dataset2, num_samples=5):
#     fig, axes = plt.subplots(2, num_samples, figsize=(10, 5))
#     for i in range(num_samples):
#         # Original image
#         img1, _ = dataset1[i]
#         axes[0, i].imshow(img1.squeeze(), cmap='gray')
#         axes[0, i].axis('off')
#         axes[0, i].set_title("Original")

#         # Rotated image
#         img2, _ = dataset2[i]
#         axes[1, i].imshow(img2.squeeze(), cmap='gray')
#         axes[1, i].axis('off')
#         axes[1, i].set_title("Rotated")

#     plt.tight_layout()
#     plt.savefig('mnist_comparison.png')

# # Call the visualization function
# visualize_samples(mnist_original, mnist_rotated)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("device: ", device)
# model.to(device)

minst_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=rotate)
train_loader = torch.utils.data.DataLoader(minst_train_dataset, batch_size=64, shuffle=True)

mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=rotate)
test_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=64, shuffle=False)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate accuracy on the training data
    model.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for images, labels in train_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train

    # Calculate accuracy on the testing data
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

# Save the trained model
model.save('artifacts/model.pt')



