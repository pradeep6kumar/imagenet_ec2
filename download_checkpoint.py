import torch
from torchvision.models import resnet50, ResNet50_Weights
import os

def download_resnet_checkpoint():
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load pretrained ResNet50
    print("Downloading pretrained ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Save the model
    checkpoint_path = os.path.join('checkpoints', 'resnet50_pretrained.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved pretrained ResNet50 to {checkpoint_path}")

if __name__ == "__main__":
    download_resnet_checkpoint() 