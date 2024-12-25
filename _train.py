import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
from PIL import Image
from model import ImageNetModel
from tqdm import tqdm
import keyboard

class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        print(f"Initializing dataset with root_dir: {root_dir}")
        
        # For training, collect images from train.X1 to train.X4 and their subfolders
        if is_train:
            print("\nLoading training data:")
            for i in range(1, 5):
                main_folder = os.path.join(root_dir, f'train.X{i}')
                print(f"Checking directory: {main_folder}")
                
                if not os.path.exists(main_folder):
                    print(f"Warning: Folder not found - {main_folder}")
                    continue
                    
                # Walk through all subfolders
                for root, dirs, files in os.walk(main_folder):
                    print(f"Scanning directory: {root}")
                    print(f"Found subdirectories: {dirs}")
                    jpeg_files = [f for f in files if f.endswith('.JPEG')]
                    if jpeg_files:
                        print(f"Found {len(jpeg_files)} images in {root}")
                        self.image_paths.extend([os.path.join(root, f) for f in jpeg_files])
        else:
            print("\nLoading validation data:")
            val_folder = os.path.join(root_dir, 'val.X')
            print(f"Checking directory: {val_folder}")
            
            if not os.path.exists(val_folder):
                print(f"Warning: Folder not found - {val_folder}")
            else:
                # Walk through all subfolders in val.X
                for root, dirs, files in os.walk(val_folder):
                    print(f"Scanning directory: {root}")
                    print(f"Found subdirectories: {dirs}")
                    jpeg_files = [f for f in files if f.endswith('.JPEG')]
                    if jpeg_files:
                        print(f"Found {len(jpeg_files)} images in {root}")
                        self.image_paths.extend([os.path.join(root, f) for f in jpeg_files])
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {root_dir}!\n"
                f"Expected structure:\n"
                f"data/\n"
                f"  ├── train.X1/\n"
                f"  │   └── subfolders with .JPEG files\n"
                f"  ├── train.X2/\n"
                f"  │   └── subfolders with .JPEG files\n"
                f"  ├── train.X3/\n"
                f"  │   └── subfolders with .JPEG files\n"
                f"  ├── train.X4/\n"
                f"  │   └── subfolders with .JPEG files\n"
                f"  └── val.X/\n"
                f"      └── subfolders with .JPEG files"
            )
        
        print(f"Total images found: {len(self.image_paths)}")
        
        # Extract labels from filenames
        for path in self.image_paths:
            # Get the parent folder name as the class label
            label = os.path.basename(os.path.dirname(path))
            self.labels.append(label)
        
        # Create label to index mapping
        unique_labels = sorted(list(set(self.labels)))
        print(f"Number of unique classes: {len(unique_labels)}")
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert string labels to indices
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class Trainer:
    def __init__(self, config):
        self.config = config
        
        # Check if data directory exists
        if not os.path.exists(config['data_dir']):
            raise ValueError(f"Data directory not found: {config['data_dir']}")
        
        print(f"\nUsing data directory: {os.path.abspath(config['data_dir'])}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_acc = 0
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        # Setup data
        self.setup_data()
        
        # Initialize model
        self.model = ImageNetModel(num_classes=100, pretrained=True).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Try to load checkpoint if exists
        self.load_checkpoint()
    
    def setup_data(self):
        # Create custom datasets
        self.train_dataset = ImageNetSubsetDataset(
            root_dir=self.config['data_dir'],
            transform=self.transform,
            is_train=True
        )
        
        self.val_dataset = ImageNetSubsetDataset(
            root_dir=self.config['data_dir'],
            transform=self.transform,
            is_train=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved new best model with accuracy: {self.best_acc:.2f}%")
    
    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'last_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Resumed training from epoch {self.start_epoch}")
            print(f"Best accuracy so far: {self.best_acc:.2f}%")
    
    def train(self):
        print(f"Training on device: {self.device}")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            print(f"\nEpoch: {epoch+1}/{self.config['epochs']}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch)
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint on keyboard interrupt
            try:
                if keyboard.is_pressed('q'):
                    print("\nTraining interrupted by user")
                    self.save_checkpoint(epoch)
                    break
            except:
                pass

if __name__ == '__main__':
    config = {
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 30,
        'num_workers': 4,
        'checkpoint_frequency': 5  # Save checkpoint every 5 epochs
    }
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    trainer = Trainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch)
        print("Checkpoint saved. Exiting...") 