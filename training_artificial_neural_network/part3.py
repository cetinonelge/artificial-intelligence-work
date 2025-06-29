# part3.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from utils import part3Plots, visualizeWeights

def load_cifar10(batch_size=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261)),
    ])
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    total_train = len(full_trainset)  # 50000
    val_count = int(0.1 * total_train)  # 10% for validation
    train_count = total_train - val_count

    train_subset, val_subset = random_split(
        full_trainset, [train_count, val_count],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def save_results(results_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)

# Architectures per homework specs (valid padding -> 0, stride=1 for conv, stride=2 for pooling)
class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 64, bias=False)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        # (3,32,32)->(16,30,30)->(8,26,26)->pool->(8,13,13)->(16,7,7)->pool->(16,3,3)->flatten(144)->fc(10)
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(16,8,kernel_size=5,stride=1,padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(8,16,kernel_size=7,stride=1,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc = nn.Linear(16*3*3, 10)

    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        # (3,32,32)->conv1->(16,30,30)->conv2->(8,28,28)->conv3->(16,24,24)->pool->(16,12,12)->conv4->(16,8,8)->pool->(16,4,4)->flatten(256)->fc(10)
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(8,16,kernel_size=5,stride=1,padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv4 = nn.Conv2d(16,16,kernel_size=5,stride=1,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc = nn.Linear(16*4*4,10)

    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN5_Debug(nn.Module):
    """
    CNN5 as specified:
    [Conv-3×3×8, ReLU,
     Conv-3×3×16, ReLU, Conv-3×3×8, ReLU, Conv-3×3×16, ReLU, MaxPool-2×2,
     Conv-3×3×16, ReLU, Conv-3×3×8, ReLU, MaxPool-2×2] + [FC10]

    Where:
    - Convolutions have kernel_size=3, stride=1, padding=0 (valid)
    - MaxPool has kernel_size=2, stride=2 (valid)
    - All activations are ReLU
    """

    def __init__(self):
        super(CNN5_Debug, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8,
                               kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8,
                               kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=8,
                               kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Based on dimension math:
        #  Input: (3,32,32)
        #  after conv1 => (8, 30,30)
        #  after conv2 => (16,28,28)
        #  after conv3 => (8,26,26)
        #  after conv4 => (16,24,24)
        #  after pool1 => (16,12,12)
        #  after conv5 => (16,10,10)
        #  after conv6 => (8, 8, 8)
        #  after pool2 => (8, 4, 4)
        #  flatten => 8 * 4 * 4 = 128
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        # Debug shape
        # print("After conv1:", x.shape)  # Should be (batch, 8, 30, 30)

        x = nn.functional.relu(self.conv2(x))
        # print("After conv2:", x.shape)  # Should be (batch, 16, 28, 28)

        x = nn.functional.relu(self.conv3(x))
        # print("After conv3:", x.shape)  # Should be (batch, 8, 26, 26)

        x = nn.functional.relu(self.conv4(x))
        # print("After conv4:", x.shape)  # Should be (batch, 16, 24, 24)

        x = self.pool1(x)
        # print("After pool1:", x.shape) # Should be (batch, 16, 12, 12)

        x = nn.functional.relu(self.conv5(x))
        # print("After conv5:", x.shape) # (batch,16,10,10)

        x = nn.functional.relu(self.conv6(x))
        # print("After conv6:", x.shape) # (batch,8,8,8)

        x = self.pool2(x)
        # print("After pool2:", x.shape) # (batch,8,4,4)

        x = x.view(x.size(0), -1)  # => (batch,128)
        # print("Flatten shape:", x.shape)

        x = self.fc(x)
        return x


models = {
    'mlp_1': MLP1(),
    'mlp_2': MLP2(),
    'cnn_3': CNN3(),
    'cnn_4': CNN4(),
    'cnn_5': CNN5_Debug()
}

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            # Move to device
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total

def main():
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, val_loader, test_loader = load_cifar10(batch_size=50)
    final_results = []
    EPOCHS = 15

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loss_curve = []
        train_acc_curve = []
        val_acc_curve = []
        step_count = 0

        for epoch in range(EPOCHS):
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                # Move inputs to device
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
                step_count += 1

                # Every 10 steps
                if step_count % 10 == 0:
                    train_loss = running_loss / 10
                    train_acc = 100.0 * correct / total
                    loss_curve.append(train_loss)
                    train_acc_curve.append(train_acc)

                    val_acc = evaluate_model(model, val_loader, device)
                    val_acc_curve.append(val_acc)

                    running_loss = 0.0

            print(f"Epoch [{epoch+1}/{EPOCHS}] : Train Acc={100.0*correct/total:.2f}%")

        test_acc = evaluate_model(model, test_loader, device)
        print(f"Test accuracy of {name}: {test_acc:.2f}%")

        if 'mlp' in name:
            weights = model.fc1.weight.data.cpu().numpy()
        else:
            weights = model.conv1.weight.data.cpu().numpy()

        result_dict = {
            'name': name,
            'loss_curve': loss_curve,
            'train_acc_curve': train_acc_curve,
            'val_acc_curve': val_acc_curve,
            'test_acc': test_acc,
            'weights': weights
        }
        save_results(result_dict, f"part3_{name}.pkl")

        # Visualize weights
        from utils import visualizeWeights  # if not imported globally
        visualizeWeights(weights, save_dir='.', filename=f'weights_{name}')

        final_results.append(result_dict)

    # Plot comparison
    from utils import part3Plots
    part3Plots(final_results, save_dir='.', filename='all_models_performance')
    print("All architectures trained; final results saved.")

if __name__ == "__main__":
    main()
