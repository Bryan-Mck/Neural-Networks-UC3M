from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image

class FashionDataset(Dataset):

    def __init__(self, csv_path, transform=None, concept_fn=None):

        self.data = pd.read_csv(csv_path)

        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.astype(np.uint8)

        self.transform = transform
        self.concept_fn = concept_fn


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        image = self.images[idx].reshape(28, 28)
        label = self.labels[idx]

        image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)

        if self.concept_fn:
            concepts = self.concept_fn(label)
            return image, concepts

        return image, label
    
class LogisticConceptDataset(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticConceptDataset, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out