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


# CNN MODEL ---------------------------------
# Define the CNN model 

class CNN(nn.Module):
    def __init__(self):
        # Calls the pytorch parent constructor
        super(CNN, self).__init__()

        # Define the convolution layers 
        # layers to identify spacial patterns
        # kernal size at 3 and low to be efficient
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # pooling layer
        # used to compress data and keep the important details in tact
        # reduces size by factor of 2
        self.pool = nn.MaxPool2d(2, 2)

        # Standard linear neural network layers (?)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    # Defines how the data is processed through the network
    def forward(self, x):

        # Conv block 1
        # Convolution first to extract features,
        # then relu to add non-linearity,
        # then pooling to reduce spatial dimensions

        # Relu is a non-linear activation function
        # The formula it uses is f(x) = max(0, x)
        # Allows to learn complex patterns
        x = self.pool(F.relu(self.conv1(x)))   # → [16, 14, 14]
        
        # Conv block 2
        # Same thing, deeper features (?)
        x = self.pool(F.relu(self.conv2(x)))   # → [32, 7, 7]
        
        # Flatten
        # This is what flatten does [batch, 32, 7, 7] → [batch, 1568]
        # reshapes
        # Connected layers need vectros not images
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

# INITALIZE MODEL ----------------------------------------------------
# Initalize the modek, the loss function, and an evaulator

# Defines the avalible hardware to train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Moves the instance of the class to hardware, either GPU or CPU
model = CNN().to(device)

# Defines the loss function
criterion = nn.CrossEntropyLoss()
# Defines the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)



# TRAINING LOOP ----------------------------------

# How many times to loop through the dataset
num_epochs = 5

# Simple loop of num_epochs
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    
    # Use train_loader for training
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")


# EVALUATION ----------------------------------

model.eval()
correct = 0
total = 0

with torch.no_grad():
    # Use test_loader for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")




# CONCEPT PREDICTOR ===============================================

# Possible attributes of labels in the dataset 
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