import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

# OTHER FILE IMPORTS
from DatasetModel import FashionDataset


# VERSION ----------------------
print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

# Creates a class to load the CSV data into an object where the labels and
# 2D arrays of pixel data are mapped together to then be put into pytorch

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

# Torch method 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    break

images.shape, labels.shape

# Each row = a Fashion MNIST class, each column = a concept
# Concepts: [has_heel, has_laces, has_sleeve, is_legwear, 
#            is_footwear, has_strap, is_outerwear, is_accessory]
# If it decides on something, it should have certain patterns as labelled in the matrix below
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


# CNN MODEL ---------------------------------
class CNN(nn.Module):
    def __init__(self):
        # Calls the pytorch parent constructor
        super(CNN, self).__init__()

        # Define the convolution layers
        # Layers to identify spatial patterns
        # Kernel size at 3 and low to be efficient
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Pooling layer
        # Used to compress data and keep the important details intact
        # Reduces size by factor of 2
        self.pool = nn.MaxPool2d(2, 2)

        # Shared backbone — compresses raw pixel features into 128 rich features
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Concept bottleneck path — 128 features → 8 concept scores
        self.concept_layer = nn.Linear(128, 8)

        # Decision layer — 8 concepts → 10 classes
        self.decision_layer = nn.Linear(8, 10)

        # Side channel path — bypasses concept bottleneck
        # 128 → 64 → 10
        self.side_fc = nn.Linear(128, 64)
        self.side_out = nn.Linear(64, 10)
        

    def forward(self, x):
        # SPATIAL FEATURE EXTRACTION
        # Conv layers scan the image with small filters looking for
        # edges, shapes, textures — things that are spatially arranged.
        # Pool halves the dimensions each time to compress the info.
        x = self.pool(F.relu(self.conv1(x)))   # [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [batch, 32, 7, 7]

        # FLATTEN
        # Conv layers produce a 3D grid. Linear layers need a 1D vector.
        # This just reshapes — no information is lost.
        x = x.view(x.size(0), -1)             # [batch, 1568]

        # SHARED BACKBONE
        # fc1 takes 1568 raw pixel-feature values and compresses them
        # into 128 richer, more abstract features.
        # ReLU keeps non-linearity so it can learn complex patterns.
        x = F.relu(self.fc1(x))               # [batch, 128]

        # Save the shared features before branching into two paths
        features = x

        # CONCEPT PATH
        # concept_layer maps those 128 features to exactly 8 scores —
        # one per concept (has_heel, has_laces, etc.)
        concept_scores = self.concept_layer(features)         # [batch, 8]

        # Sigmoid squashes each score to between 0 and 1.
        # This is what makes them interpretable as probabilities —
        # 0.9 means "very likely has this concept", 0.1 means "probably not".
        concept_preds = torch.sigmoid(concept_scores)         # [batch, 8]

        # Decision layer — only sees concepts, nothing else
        # This is f(c) from the formula
        fc_logits = self.decision_layer(concept_preds)        # [batch, 10]

        # SIDE CHANNEL PATH
        # Takes the same 128 features as the concept path but bypasses
        # the concept bottleneck entirely — goes directly to class logits.
        # This is s(x) from the formula
        sc_x = F.relu(self.side_fc(features))                 # [batch, 64]
        sc_logits = self.side_out(sc_x)                       # [batch, 10]

        # Return all three — concept_preds for concept loss,
        # fc_logits and sc_logits to be combined in the training loop
        return concept_preds, fc_logits, sc_logits


def RunCBM():

    # INITIALIZE MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Moves the instance of the class to hardware, either GPU or CPU
    model = CNN().to(device)

    # Move concept targets to same device as model
    concept_targets_all = CLASS_TO_CONCEPTS.to(device)

    # Defines the loss function
    criterion = nn.CrossEntropyLoss()
    # Defines the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # TRAINING LOOP
    num_epochs = 5

    for epoch in range(num_epochs):
        # Always call model.train() so it is in training mode
        # This matters for certain layers like dropout or batchnorm,
        # which behave differently during training vs. evaluation.
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Look up concept targets for this batch's class labels
            concept_targets = concept_targets_all[labels]     # [batch, 8]

            # Forward pass — returns three things now
            concept_preds, fc_logits, sc_logits = model(images)

            # Combine both paths — this is f(c) + s(x)
            final_logits = fc_logits + sc_logits

            # Loss 1: how well did it predict the concepts?
            concept_loss = F.binary_cross_entropy(concept_preds, concept_targets)

            # Loss 2: how well did it predict the final class?
            class_loss = criterion(final_logits, labels)

            # Combined loss — alpha controls the balance between concept and class loss
            alpha = 0.5
            loss = alpha * concept_loss + (1 - alpha) * class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


    # EVALUATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Unpack three values and combine for final prediction
            concept_preds, fc_logits, sc_logits = model(images)
            final_logits = fc_logits + sc_logits
            _, predicted = torch.max(final_logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

RunCBM()