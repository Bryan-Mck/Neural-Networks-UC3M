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
from sklearn.metrics import roc_auc_score, f1_score

# OTHER FILE IMPORTS
from DatasetModel import FashionDataset


# VERSION ----------------------
print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FashionDataset(
    csv_path='C:\\Users\\bryan\\OneDrive\\Documents\\coding\\python\\Neural\\project1\\Dataset\\fashion-mnist_train.csv',
    transform=transform
)

test_dataset = FashionDataset(
    csv_path='C:\\Users\\bryan\\OneDrive\\Documents\\coding\\python\\Neural\\project1\\Dataset\\fashion-mnist_test.csv',
    transform=transform
)

print(len(train_dataset))
image, label = train_dataset[6000]
print(label)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

CLASS_TO_CONCEPTS = torch.tensor([
    [0, 0, 1, 0, 0, 0, 0, 0],  # 0: T-shirt
    [0, 0, 0, 1, 0, 0, 0, 0],  # 1: Trouser
    [0, 0, 1, 0, 0, 0, 0, 0],  # 2: Pullover
    [0, 0, 1, 0, 0, 0, 0, 0],  # 3: Dress
    [0, 0, 1, 0, 0, 0, 1, 0],  # 4: Coat
    [0, 0, 0, 0, 1, 1, 0, 0],  # 5: Sandal
    [0, 0, 1, 0, 0, 0, 0, 0],  # 6: Shirt
    [0, 1, 0, 0, 1, 0, 0, 0],  # 7: Sneaker
    [0, 0, 0, 0, 0, 1, 0, 1],  # 8: Bag
    [1, 0, 0, 0, 1, 0, 0, 0],  # 9: Ankle boot
], dtype=torch.float32)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.concept_layer = nn.Linear(128, 8)
        self.decision_layer = nn.Linear(8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        concept_scores = self.concept_layer(x)
        concept_preds = torch.sigmoid(concept_scores)
        class_logits = self.decision_layer(concept_preds)
        return concept_preds, class_logits


def RunCBM():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    concept_targets_all = CLASS_TO_CONCEPTS.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TRAINING LOOP ----------------------------------

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            concept_targets = concept_targets_all[labels]
            concept_preds, class_logits = model(images)

            concept_loss = F.binary_cross_entropy(concept_preds, concept_targets)
            class_loss = criterion(class_logits, labels)

            alpha = 0.5
            loss = alpha * concept_loss + (1 - alpha) * class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


    # EVALUATION ----------------------------------

    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    all_concept_preds = []
    all_concept_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            concept_preds, class_logits = model(images)
            probs = F.softmax(class_logits, dim=1)

            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect concept predictions and targets for F1
            concept_binary = (concept_preds > 0.5).cpu().numpy()
            concept_targets = concept_targets_all[labels].cpu().numpy()
            all_concept_preds.append(concept_binary)
            all_concept_targets.append(concept_targets)

    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_targets = np.vstack(all_concept_targets)

    # CLASS METRICS
    accuracy = 100 * correct / total
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f"\nClass Prediction Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUROC:    {auroc:.4f}")

    # CONCEPT METRICS
    f1_per_concept = f1_score(all_concept_targets, all_concept_preds, average=None)
    macro_f1 = f1_score(all_concept_targets, all_concept_preds, average='macro')

    concept_correct = torch.tensor(
        (all_concept_preds == all_concept_targets).sum(axis=0), dtype=torch.float32
    )
    accuracy_per_concept = concept_correct / total

    print(f"\nConcept Prediction Results:")
    print(f"{'Concept':<30} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 52)
    for i, name in enumerate(concept_names):
        print(f"{name:<30} {accuracy_per_concept[i].item():>10.4f} {f1_per_concept[i]:>10.4f}")
    print("-" * 52)
    print(f"{'Macro Average':<30} {accuracy_per_concept.mean().item():>10.4f} {macro_f1:>10.4f}")

RunCBM()