import torch
import torch.nn as nn
import torchvision.models as tv_models
 
 # Simple CNN with 3 layers
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # This just gets the features from the 32x32 image
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     
            nn.Dropout2d(0.2),
 
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        
            nn.Dropout2d(0.3),
 
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       
            nn.Dropout2d(0.4),
        )
        # Since there are 10 classes, this will help decide
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
 
 # Less code than SmallCNN because the layout of Resnet is already
 # Set up in a models function
def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    # Replace stem: original stride-2, 7×7 → stride-1, 3×3
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()          # remove the early max-pool
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
 