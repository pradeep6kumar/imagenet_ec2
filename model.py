import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os

class ImageNetModel(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ImageNetModel, self).__init__()
        
        # Initialize ResNet50 without pretrained weights
        print("Initializing ResNet50 from scratch...")
        self.model = resnet50(weights=None)
        
        # Get the input features of final layer
        num_features = self.model.fc.in_features
        
        # Replace the final layer with proper initialization
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(num_features, num_classes),
            nn.BatchNorm1d(num_classes)  # Add batch norm for stability
        )
        
        # Initialize the new layers properly
        nn.init.xavier_uniform_(self.model.fc[1].weight)
        nn.init.zeros_(self.model.fc[1].bias)
        
        # Enable memory efficient features
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    def forward(self, x):
        logits = self.model(x)
        return logits  # Make sure we return logits for proper loss calculation 