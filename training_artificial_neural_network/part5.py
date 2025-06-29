# part5.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from utils import part5Plots

########################################################
# 1) DATA LOADING
########################################################
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

    total_train = len(full_trainset)  # 50,000
    val_count = int(0.1 * total_train)
    train_count = total_train - val_count

    gen = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_trainset,
                                           [train_count, val_count],
                                           generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

########################################################
# 2) ARCHITECTURE (CNN3)
########################################################
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        # [Conv-3×3×16, ReLU,
        #  Conv-5×5×8, ReLU, MaxPool-2×2,
        #  Conv-7×7×16, MaxPool-2×2] + FC10
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(16,8,kernel_size=5,stride=1,padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(8,16,kernel_size=7,stride=1,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # dimension => (16,3,3) => flatten=144
        self.fc = nn.Linear(16*3*3, 10)

    def forward(self,x):
        x = torch.relu(self.conv1(x))   # (16,30,30)
        x = torch.relu(self.conv2(x))   # (8,26,26)
        x = self.pool1(x)              # (8,13,13)
        x = torch.relu(self.conv3(x))   # (16,7,7)
        x = self.pool2(x)              # (16,3,3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

########################################################
# 3) EVALUATION
########################################################
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total

########################################################
# 4) TRAIN FUNCTION
########################################################
def train_model(lr, train_loader, val_loader, model, device, epochs=20):
    """
    Trains with:
     - SGD(lr=lr, momentum=0.0)
     - batch_size=50
    Records training loss + validation accuracy every 10 steps
    Returns (loss_curve, val_acc_curve)
    """
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    loss_curve = []
    val_acc_curve = []
    step_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for (x, y) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (LR={lr})"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            step_count += 1
            running_loss += loss.item()

            # record every 10 steps
            if step_count % 10 == 0:
                avg_loss = running_loss / 10
                loss_curve.append(avg_loss)
                running_loss = 0.0
                val_acc = evaluate(model, val_loader, device)
                val_acc_curve.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] LR={lr} finished.")

    return loss_curve, val_acc_curve

########################################################
# 5) MAIN
########################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 5a) Load data
    train_loader, val_loader, test_loader = load_cifar10(batch_size=50)
    EPOCHS = 20

    # 5b) Instantiate CNN3 for each LR
    model_1 = CNN3()
    model_01 = CNN3()
    model_001 = CNN3()

    print("\n--- Training with LR=0.1 ---")
    loss_curve_1, val_acc_curve_1 = train_model(0.1, train_loader, val_loader, model_1, device, EPOCHS)

    print("\n--- Training with LR=0.01 ---")
    loss_curve_01, val_acc_curve_01 = train_model(0.01, train_loader, val_loader, model_01, device, EPOCHS)

    print("\n--- Training with LR=0.001 ---")
    loss_curve_001, val_acc_curve_001 = train_model(0.001, train_loader, val_loader, model_001, device, EPOCHS)

    # Create dictionary
    result_dict = {
        'name': 'cnn3',
        'loss_curve_1': loss_curve_1,
        'loss_curve_01': loss_curve_01,
        'loss_curve_001': loss_curve_001,
        'val_acc_curve_1': val_acc_curve_1,
        'val_acc_curve_01': val_acc_curve_01,
        'val_acc_curve_001': val_acc_curve_001
    }

    # Save
    with open("part5_cnn3.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    # Plot
    part5Plots(result_dict, save_dir='.', filename='part5_lr_comparison')
    print("\nPart5 - LR experiment (0.1, 0.01, 0.001) completed. Plots saved.")

    # ---------------------------------------------------------
    # 5c) Manual scheduled LR approach
    # ---------------------------------------------------------
    print("\n--- Scheduled Learning Rate Approach ---")

    scheduled_model = CNN3().to(device)
    optimizer_sch = optim.SGD(scheduled_model.parameters(), lr=0.1, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # We'll store only the validation accuracy
    scheduled_val_acc = []

    # first "plateau" ~ epoch=5 (as an example)
    plateau_epoch_1 = 5
    for epoch in range(plateau_epoch_1):
        scheduled_model.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_sch.zero_grad()
            outputs = scheduled_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer_sch.step()
        val_acc_sch = evaluate(scheduled_model, val_loader, device)
        scheduled_val_acc.append(val_acc_sch)
        print(f"Scheduled LR=0.1, Epoch {epoch+1}/{plateau_epoch_1}, Val Acc={val_acc_sch:.2f}%")

    # now set LR=0.01
    for g in optimizer_sch.param_groups:
        g['lr'] = 0.01

    # continue training until ~ epoch=15
    plateau_epoch_2 = 15
    for epoch in range(plateau_epoch_1, plateau_epoch_2):
        scheduled_model.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_sch.zero_grad()
            outputs = scheduled_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer_sch.step()
        val_acc_sch = evaluate(scheduled_model, val_loader, device)
        scheduled_val_acc.append(val_acc_sch)
        print(f"Scheduled LR=0.01, Epoch {epoch+1}/{plateau_epoch_2}, Val Acc={val_acc_sch:.2f}%")

    # set LR=0.001
    for g in optimizer_sch.param_groups:
        g['lr'] = 0.001

    # train until epoch=30
    for epoch in range(plateau_epoch_2, 30):
        scheduled_model.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_sch.zero_grad()
            outputs = scheduled_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer_sch.step()
        val_acc_sch = evaluate(scheduled_model, val_loader, device)
        scheduled_val_acc.append(val_acc_sch)
        print(f"Scheduled LR=0.001, Epoch {epoch+1}/30, Val Acc={val_acc_sch:.2f}%")

    final_test_acc_sch = evaluate(scheduled_model, test_loader, device)
    print(f"Final Test Accuracy with scheduled LR approach: {final_test_acc_sch:.2f}%")

    print("Part5 scheduled LR approach done.")

if __name__ == "__main__":
    main()
