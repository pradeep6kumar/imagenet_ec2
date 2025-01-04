import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import logging

logger = logging.getLogger(__name__)

class ImageNetModel(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ImageNetModel, self).__init__()
        
        # Initialize ResNet50 from scratch
        logger.info("Initializing ResNet50 from scratch...")
        self.model = resnet50(weights=None)  # Always initialize without weights
        
        # Modify the final layer for ImageNet (1000 classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Initialize the new fc layer
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        # Remove the final layer to get features
        return torch.flatten(self.model.avgpool(self.model.layer4(
            self.model.layer3(self.model.layer2(
                self.model.layer1(self.model.maxpool(
                    self.model.relu(self.model.bn1(
                        self.model.conv1(x))))))))), 1) 