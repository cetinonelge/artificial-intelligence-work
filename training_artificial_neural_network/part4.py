import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# Provided utilities
from utils import part4Plots

#########################
# 1) DATA PREPARATION
#########################

def load_cifar10(batch_size=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261)),
    ])
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 10% validation, 90% training
    total_train = len(full_trainset)  # 50,000
    val_count = int(0.1 * total_train)
    train_count = total_train - val_count

    train_subset, val_subset = random_split(
        full_trainset, [train_count, val_count],
        generator=torch.Generator().manual_seed(42)
    )

    # shuffle=True for training set
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader

#########################
# 2) ARCHITECTURE DEFINITIONS
#########################

# We'll define versions of each architecture that let us pick ReLU or Sigmoid
# for consistency with question 3.1, we keep them minimal

class MLP1(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 32)
        self.fc2 = nn.Linear(32, 10)
        self.activation_type = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.fc1(x))
        else:
            x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP2(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 32)
        self.fc2 = nn.Linear(32, 64, bias=False)
        self.fc3 = nn.Linear(64, 10)
        self.activation_type = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
        else:
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN3(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=7, padding=0)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16*3*3, 10)
        self.activation_type = activation

    def forward(self, x):
        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.conv1(x))
            x = torch.sigmoid(self.conv2(x))
        else:
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.conv3(x))
        else:
            x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN4(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=0)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=0)
        self.conv3 = nn.Conv2d(8,16,kernel_size=5,padding=0)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(16,16,kernel_size=5,padding=0)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16*4*4,10)
        self.activation_type = activation

    def forward(self, x):
        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.conv1(x))
            x = torch.sigmoid(self.conv2(x))
            x = torch.sigmoid(self.conv3(x))
        else:
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.relu(self.conv3(x))
        x = self.pool1(x)

        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.conv4(x))
        else:
            x = nn.functional.relu(self.conv4(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN5(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,padding=0)
        self.conv2 = nn.Conv2d(8,16,kernel_size=3,padding=0)
        self.conv3 = nn.Conv2d(16,8,kernel_size=3,padding=0)
        self.conv4 = nn.Conv2d(8,16,kernel_size=3,padding=0)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(16,16,kernel_size=3,padding=0)
        self.conv6 = nn.Conv2d(16,8,kernel_size=3,padding=0)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(8*4*4,10)
        self.activation_type = activation

    def forward(self, x):
        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.conv1(x))
            x = torch.sigmoid(self.conv2(x))
            x = torch.sigmoid(self.conv3(x))
            x = torch.sigmoid(self.conv4(x))
        else:
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.relu(self.conv3(x))
            x = nn.functional.relu(self.conv4(x))
        x = self.pool1(x)

        if self.activation_type == 'sigmoid':
            x = torch.sigmoid(self.conv5(x))
            x = torch.sigmoid(self.conv6(x))
        else:
            x = nn.functional.relu(self.conv5(x))
            x = nn.functional.relu(self.conv6(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#########################
# 3) TRAIN/EVAL for PART 4
#########################
def compute_gradient_magnitude(layer):
    """
    Return the norm of the gradient of the given layer's weights as a single float.
    If there's no gradient yet, returns 0.
    """
    if layer.weight.grad is None:
        return 0.0
    return layer.weight.grad.data.norm().item()

def train_activation_model(model, train_loader, first_layer, epochs=15):
    """
    Trains a model using:
      - SGD with lr=0.01, momentum=0.0
      - batch_size=50
    Records:
      - 'loss_curve' every 10 steps
      - 'grad_curve' every 10 steps (magnitude of grad on first_layer)
    Returns (loss_curve, grad_curve).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

    loss_curve = []
    grad_curve = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        for (x, y) in train_loader:
            step_count += 1
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            # record every 10 steps
            if step_count % 10 == 0:
                loss_curve.append(loss.item())
                grad_magnitude = compute_gradient_magnitude(first_layer)
                grad_curve.append(grad_magnitude)
            optimizer.step()

    return loss_curve, grad_curve

def part4_experiment(arch_name, model_relu, model_sigmoid, first_layer_relu, first_layer_sigmoid,
                     train_loader, epochs=15):
    """
    Trains one architecture with ReLU and Sigmoid, returns dictionary with:
      'name', 'relu_loss_curve', 'sigmoid_loss_curve',
              'relu_grad_curve', 'sigmoid_grad_curve'
    """
    # Train ReLU
    print(f"Training {arch_name} - ReLU version")
    relu_loss_curve, relu_grad_curve = train_activation_model(model_relu, train_loader,
                                                              first_layer_relu, epochs=epochs)

    # Train Sigmoid
    print(f"Training {arch_name} - Sigmoid version")
    sigmoid_loss_curve, sigmoid_grad_curve = train_activation_model(model_sigmoid, train_loader,
                                                                    first_layer_sigmoid, epochs=epochs)

    # Create dictionary
    result_dict = {
        'name': arch_name,
        'relu_loss_curve': relu_loss_curve,
        'sigmoid_loss_curve': sigmoid_loss_curve,
        'relu_grad_curve': relu_grad_curve,
        'sigmoid_grad_curve': sigmoid_grad_curve
    }
    return result_dict

#########################
# 4) MAIN
#########################
def main():
    # Load CIFAR10
    train_loader, val_loader, test_loader = load_cifar10(batch_size=50)
    print("Loaded CIFAR-10 with batch size=50 for Part 4...")

    # We'll do 15 epochs as stated
    EPOCHS = 15

    # Arch definitions for ReLU vs Sigmoid
    archs = {
      'mlp_1': (MLP1('relu'), MLP1('sigmoid')),
      'mlp_2': (MLP2('relu'), MLP2('sigmoid')),
      'cnn_3': (CNN3('relu'), CNN3('sigmoid')),
      'cnn_4': (CNN4('relu'), CNN4('sigmoid')),
      'cnn_5': (CNN5('relu'), CNN5('sigmoid')),
    }

    results = []

    for arch_name, (model_relu, model_sigmoid) in archs.items():
        print(f"\n=== Running Part4 experiment for {arch_name} ===")

        # Identify the first layer
        # For MLP: first layer is fc1
        # For CNN: first layer is conv1
        if 'mlp' in arch_name:
            first_layer_relu = model_relu.fc1
            first_layer_sigmoid = model_sigmoid.fc1
        else:
            first_layer_relu = model_relu.conv1
            first_layer_sigmoid = model_sigmoid.conv1

        # part4_experiment
        arch_dict = part4_experiment(arch_name,
                                     model_relu,
                                     model_sigmoid,
                                     first_layer_relu,
                                     first_layer_sigmoid,
                                     train_loader,
                                     epochs=EPOCHS)
        # save the dictionary
        with open(f'part4_{arch_name}.pkl', 'wb') as f:
            pickle.dump(arch_dict, f)
        results.append(arch_dict)

    # Now we use part4Plots
    print("\nGenerating Part4 performance comparison plots...")
    part4Plots(results, save_dir='.', filename='part4_performance')

    print("Part4 experiment completed! All results saved and plot generated.")

if __name__ == "__main__":
    main()
