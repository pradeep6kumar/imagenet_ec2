import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os

class ImageNetModel(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ImageNetModel, self).__init__()
        
        # Load ResNet50 with memory efficient settings
        self.model = resnet50(weights=None)
        
        if pretrained:
            checkpoint_path = os.path.join('checkpoints', 'resnet50_pretrained.pth')
            if os.path.exists(checkpoint_path):
                print(f"Loading pretrained weights from {checkpoint_path}")
                self.model.load_state_dict(torch.load(checkpoint_path))
            else:
                print("No local checkpoint found, downloading pretrained weights...")
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Enable memory efficient features
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    def forward(self, x):
        return self.model(x) 