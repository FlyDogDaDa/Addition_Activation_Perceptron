import AdditionActivationPerceptron as AAP
import torch
import json
import os
from torch import nn as nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class MnistCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 28x28
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2),
            # 14x14
            nn.Conv2d(4, 8, kernel_size=3),  # 12x12
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            # 6x6
            nn.Flatten(),
            # 36
            AAP.Linear(288, 128),
            nn.BatchNorm1d(128),
            AAP.Linear(128, 64),
            nn.BatchNorm1d(64),
            AAP.Linear(64, 32),
            nn.BatchNorm1d(32),
            AAP.Linear(32, 16),
            nn.BatchNorm1d(16),
            AAP.Linear(16, 10),
            nn.BatchNorm1d(10),
        )

    def forward(self, x):
        return self.model(x)


epoch = 20
# 創建資料夾
root = "models"
os.makedirs(root, exist_ok=True)

# 訓練程式
model = MnistCnn()
model.cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Split the dataset into training and testing sets
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# Define data loaders
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)


def test_model():
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            correct += loss.item()
    test_loss_list.append(correct / len(test_loader))


train_loss_list = []
test_loss_list = []
# 訓練模型
for e in range(epoch):
    test_model()
    if not (e % 5):
        torch.save(model, os.path.join(root, f"mnist_cnn_{e}e.pt"))
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "epoch [{}/{}], step [{}/{}], loss: {:.4f}".format(
                    e + 1, 20, i + 1, len(train_loader), loss.item()
                )
            )
    train_loss_list.append(running_loss / len(train_loader))
test_model()

with open(os.path.join(root, "loss.json"), "w") as f:
    json.dump({"train_loss": train_loss_list, "test_loss": test_loss_list}, f)
