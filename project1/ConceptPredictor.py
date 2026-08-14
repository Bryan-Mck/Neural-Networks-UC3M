import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

# OTHER FILE IMPORTS
from DatasetModel import FashionDataset

# CONCEPT PREDICTOR ===============================================

concept_names = [
    "is_footwear",
    "is_closed_footwear",
    "is_footwear_or_bag",
    "has_sleeves",
    "has_collar",
    "is_long_garment",
    "is_outerwear_layer",
    "is_legwear_or_footwear"
]

concept_map = {
    "is_footwear": [5,7,9],
    "is_closed_footwear": [7,9],
    "is_footwear_or_bag": [5,7,8,9],
    "has_sleeves": [0,2,3,4,6],
    "has_collar": [4,6],
    "is_long_garment": [3,4],
    "is_outerwear_layer": [2,4],
    "is_legwear_or_footwear": [1,5,7,9]
}

def label_to_concepts(label):
    return torch.tensor([
        label in [5,7,9],
        label in [7,9],
        label in [5,7,8,9],
        label in [0,2,3,4,6],
        label in [4,6],
        label in [3,4],
        label in [2,4],
        label in [1,5,7,9]
    ], dtype=torch.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FashionDataset(
    csv_path='C:\\Users\\bryan\\OneDrive\\Documents\\coding\\python\\Neural\\project1\\Dataset\\fashion-mnist_train.csv',
    transform=transform,
    concept_fn=label_to_concepts,
)

test_dataset = FashionDataset(
    csv_path='C:\\Users\\bryan\\OneDrive\\Documents\\coding\\python\\Neural\\project1\\Dataset\\fashion-mnist_test.csv',
    transform=transform,
    concept_fn=label_to_concepts,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    break

images.shape, labels.shape

def RunCP():

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

            self.pool = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 8)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # TRAINING LOOP ----------------------------------

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, concepts in train_loader:
            images, concepts = images.to(device), concepts.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, concepts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


    # EVALUATION ----------------------------------

    model.eval()
    correct = torch.zeros(8)
    total = torch.zeros(8)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, concepts in test_loader:
            images, concepts = images.to(device), concepts.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            correct += (preds == concepts.bool()).sum(dim=0).cpu()
            total += concepts.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(concepts.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    accuracy_per_concept = correct / total
    macro_accuracy = accuracy_per_concept.mean()

    # F1 per concept
    f1_per_concept = f1_score(all_targets, all_preds, average=None)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    # Print results as a table
    print("\nConcept Predictor Results:")
    print(f"{'Concept':<30} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 52)
    for i, name in enumerate(concept_names):
        print(f"{name:<30} {accuracy_per_concept[i].item():>10.4f} {f1_per_concept[i]:>10.4f}")
    print("-" * 52)
    print(f"{'Macro Average':<30} {macro_accuracy.item():>10.4f} {macro_f1:>10.4f}")

RunCP()