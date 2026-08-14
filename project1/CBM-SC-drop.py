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
from Steerability_test import run_steerability_experiment, plot_steerability, print_concept_ranking


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

for images, labels in train_loader:
    break

images.shape, labels.shape

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


class CNN(nn.Module):
    def __init__(self, side_dropout_p=0.0):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.concept_layer = nn.Linear(128, 8)
        self.decision_layer = nn.Linear(8, 10)
        self.side_fc = nn.Linear(128, 64)
        self.side_out = nn.Linear(64, 10)
        self.side_dropout = nn.Dropout(p=side_dropout_p)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        features = x

        concept_scores = self.concept_layer(features)
        concept_preds = torch.sigmoid(concept_scores)
        fc_logits = self.decision_layer(concept_preds)

        sc_x = F.relu(self.side_fc(features))
        sc_x = self.side_dropout(sc_x)
        sc_logits = self.side_out(sc_x)

        return concept_preds, fc_logits, sc_logits


def train_and_evaluate(dropout_p, device):

    model = CNN(side_dropout_p=dropout_p).to(device)
    concept_targets_all = CLASS_TO_CONCEPTS.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            concept_targets = concept_targets_all[labels]
            concept_preds, fc_logits, sc_logits = model(images)
            final_logits = fc_logits + sc_logits

            concept_loss = F.binary_cross_entropy(concept_preds, concept_targets)
            class_loss = criterion(final_logits, labels)

            alpha = 0.5
            loss = alpha * concept_loss + (1 - alpha) * class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"  [p={dropout_p}] Epoch {epoch+1}, Loss: {running_loss:.4f}")

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
            concept_preds, fc_logits, sc_logits = model(images)
            final_logits = fc_logits + sc_logits
            probs = F.softmax(final_logits, dim=1)

            _, predicted = torch.max(final_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            concept_binary = (concept_preds > 0.5).cpu().numpy()
            concept_targets = concept_targets_all[labels].cpu().numpy()
            all_concept_preds.append(concept_binary)
            all_concept_targets.append(concept_targets)

    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_targets = np.vstack(all_concept_targets)

    accuracy = 100 * correct / total
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    macro_f1 = f1_score(all_concept_targets, all_concept_preds, average='macro')
    f1_per_concept = f1_score(all_concept_targets, all_concept_preds, average=None)

    return model, accuracy, auroc, macro_f1, f1_per_concept


def RunDropoutSweep():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    accuracies = []
    aurocs = []
    macro_f1s = []
    all_f1s = []
    models = {}

    for p in dropout_values:
        print(f"\nTraining with dropout p={p}...")
        model, acc, auc, mf1, f1s = train_and_evaluate(p, device)
        accuracies.append(acc)
        aurocs.append(auc)
        macro_f1s.append(mf1)
        all_f1s.append(f1s)
        models[p] = model

    # PRINT RESULTS TABLE
    print("\nDropout Sweep Results:")
    print(f"{'Dropout':<10} {'Accuracy':>10} {'AUROC':>10} {'Macro F1':>10}")
    print("-" * 42)
    for i, p in enumerate(dropout_values):
        print(f"{p:<10} {accuracies[i]:>10.2f}% {aurocs[i]:>10.4f} {macro_f1s[i]:>10.4f}")

    # PLOT
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Side-Channel Dropout Probability")
    ax1.set_ylabel("Accuracy (%)", color="blue")
    ax1.plot(dropout_values, accuracies, marker='o', color="blue", label="Accuracy")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("AUROC", color="red")
    ax2.plot(dropout_values, aurocs, marker='s', color="red", label="AUROC")
    ax2.tick_params(axis='y', labelcolor="red")

    fig.suptitle("Hybrid CBM: Accuracy and AUROC vs Side-Channel Dropout")
    fig.tight_layout()
    plt.savefig("dropout_sweep.png", dpi=150)
    plt.show()
    print("Plot saved as dropout_sweep.png")

    # CONCEPT F1 PER DROPOUT
    print("\nConcept Macro F1 per Dropout Level:")
    print(f"{'Concept':<30}", end="")
    for p in dropout_values:
        print(f"  p={p}", end="")
    print()
    print("-" * 80)
    for i, name in enumerate(concept_names):
        print(f"{name:<30}", end="")
        for j in range(len(dropout_values)):
            print(f"  {all_f1s[j][i]:.4f}", end="")
        print()

    return models, device


models, device = RunDropoutSweep()


print("\nRunning steerability experiment on p=0.0 model")
base_model = models[0.0]

results = run_steerability_experiment(base_model, test_loader, device, num_batches=50)
print_concept_ranking(results)
plot_steerability(results, title="Hybrid CBM — Concept Intervention Effects")