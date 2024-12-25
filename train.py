import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import asyncio
import threading
from PIL import Image
from model import ImageNetModel
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchvision.datasets import ImageFolder
from logging.handlers import RotatingFileHandler
import time

# Add rotating file handler
rotating_handler = RotatingFileHandler("training.log", maxBytes=5_000_000, backupCount=5)  # 5 MB per log

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.addHandler(rotating_handler)

class ImageNetSubsetDataset(Dataset):

    def __init__(self, root_dir, transform=None, is_train=True):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        logger.info(f"Initializing dataset with root_dir: {root_dir}")
        
        # For training, collect images from train.X1 to train.X4 and their subfolders
        if is_train:
            logger.info("Loading training data:")
            for i in range(1, 5):
                main_folder = os.path.join(root_dir, f'train.X{i}')
                logger.info(f"Checking directory: {main_folder}")
                
                if not os.path.exists(main_folder):
                    logger.warning(f"Folder not found - {main_folder}")
                    continue
                    
                # Walk through all subfolders
                for root, dirs, files in os.walk(main_folder):
                    logger.info(f"Scanning directory: {root}")
                    jpeg_files = [f for f in files if f.endswith('.JPEG')]
                    if jpeg_files:
                        logger.info(f"Found {len(jpeg_files)} images in {root}")
                        self.image_paths.extend([os.path.join(root, f) for f in jpeg_files])
        else:
            logger.info("Loading validation data:")
            val_folder = os.path.join(root_dir, 'val.X')
            logger.info(f"Checking directory: {val_folder}")
            
            if not os.path.exists(val_folder):
                logger.warning(f"Folder not found - {val_folder}")
            else:
                # Walk through all subfolders in val.X
                for root, dirs, files in os.walk(val_folder):
                    logger.info(f"Scanning directory: {root}")
                    jpeg_files = [f for f in files if f.endswith('.JPEG')]
                    if jpeg_files:
                        logger.info(f"Found {len(jpeg_files)} images in {root}")
                        self.image_paths.extend([os.path.join(root, f) for f in jpeg_files])
        
        if len(self.image_paths) == 0:
            logger.error("No images found in the dataset!")
            raise ValueError("Dataset is empty.")

        logger.info(f"Total images found: {len(self.image_paths)}")
        
        # Extract labels from filenames
        for path in self.image_paths:
            label = os.path.basename(os.path.dirname(path))
            self.labels.append(label)
        
        unique_labels = sorted(list(set(self.labels)))
        logger.info(f"Number of unique classes: {len(unique_labels)}")
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in self.labels]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        max_retries = 3
        for retry in range(max_retries):
            try:
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert('RGB')
                label = self.labels[idx]

                if self.transform:
                    image = self.transform(image)

                return image, label
            except Exception as e:
                if retry == max_retries - 1:
                    logger.error(f"Failed to load image {image_path} after {max_retries} attempts: {e}")
                    raise
                time.sleep(0.1)

class Trainer:
    
    def __init__(self, config):
        self.config = config

        # Check if data directory exists
        if not os.path.exists(config['data_dir']):
            logger.error(f"Data directory not found: {config['data_dir']}")
            raise ValueError("Data directory does not exist.")

        logger.info(f"Using data directory: {os.path.abspath(config['data_dir'])}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_acc = 0

        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize model
        self.model = ImageNetModel(num_classes=100, pretrained=True).to(self.device)

        # Loss and optimizer with simple parameters
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize basic SGD optimizer without momentum
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )

        # Setup data first to get loader length
        self.setup_data()

        # Use a simpler scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )

        # Initialize async checkpoint saving
        self.checkpoint_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        self.checkpoint_worker = threading.Thread(
            target=self._run_checkpoint_worker, 
            daemon=True
        )
        self.checkpoint_worker.start()

        # Load checkpoint if exists
        self.load_checkpoint()

        # Training parameters
        self.grad_clip_value = 1.0
        self.patience = 10
        self.early_stopping_counter = 0
        self.min_delta = 0.001

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def _run_checkpoint_worker(self):
        """Run the checkpoint worker in its own thread with its own event loop"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.checkpoint_saving_worker())

    async def checkpoint_saving_worker(self):
        while True:
            try:
                checkpoint_data, path = await self.checkpoint_queue.get()
                if checkpoint_data is None:  # Shutdown signal
                    logger.info("Checkpoint saving worker shutting down.")
                    break
                
                # Save checkpoint in a separate thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: torch.save(checkpoint_data, path)
                )
                logger.info(f"Checkpoint saved to {path}")
                
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
            finally:
                self.checkpoint_queue.task_done()

    def setup_data(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Use ImageNetSubsetDataset instead of ImageFolder
        self.train_dataset = ImageNetSubsetDataset(
            root_dir=self.config['data_dir'],
            transform=train_transform,
            is_train=True
        )
        
        self.val_dataset = ImageNetSubsetDataset(
            root_dir=self.config['data_dir'],
            transform=val_transform,
            is_train=False
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            prefetch_factor=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            prefetch_factor=2
        )


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        scaler = GradScaler()

        logger.info("Starting training epoch...")
        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader)):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        torch.cuda.empty_cache()  # Clear GPU cache before validation
        
        total_loss = 0
        correct = 0
        total = 0
        top5_correct = 0

        logger.info("Starting validation...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.topk(5, dim=1)  # Top-5 predictions
                total += labels.size(0)
                correct += (predicted[:, 0] == labels).sum().item()
                top5_correct += torch.sum(torch.any(predicted == labels.unsqueeze(1), dim=1)).item()

                if batch_idx % 100 == 0:
                    logger.info(f"Validation Progress: Batch {batch_idx}/{len(self.val_loader)}")

        top5_acc = 100. * top5_correct / total
        logger.info(f"Validation completed. Avg Loss = {total_loss / len(self.val_loader):.4f}, Top-1 Accuracy = {100. * correct / total:.2f}%, Top-5 Accuracy = {top5_acc:.2f}%")
        torch.cuda.empty_cache()  # Add memory cleanup after validation
        return total_loss / len(self.val_loader), 100. * correct / total


    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config
        }

        # Use the stored event loop
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'last_checkpoint.pth')
        asyncio.run_coroutine_threadsafe(
            self.checkpoint_queue.put((checkpoint, checkpoint_path)), 
            self.loop
        )

        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            asyncio.run_coroutine_threadsafe(
                self.checkpoint_queue.put((checkpoint, best_path)), 
                self.loop
            )
    
    def shutdown_worker(self):
        """Properly shutdown the checkpoint worker"""
        logger.info("Initiating shutdown sequence...")
        try:
            # Send shutdown signal to worker
            asyncio.run_coroutine_threadsafe(
                self.checkpoint_queue.put((None, None)), 
                self.loop
            )
            # Wait for worker to finish with timeout
            self.checkpoint_worker.join(timeout=10)
            
            # Force cleanup if worker doesn't respond
            if self.checkpoint_worker.is_alive():
                logger.warning("Checkpoint worker didn't shut down gracefully, forcing shutdown")
            
            self.loop.stop()
            self.loop.close()
            logger.info("Checkpoint worker has shut down.")
        except Exception as e:
            logger.error(f"Error during worker shutdown: {e}")



    
    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'last_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Resumed training from epoch {self.start_epoch}")
                logger.info(f"Best accuracy so far: {self.best_acc:.2f}%")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
        else:
            logger.info("No existing checkpoint found. Starting training from scratch.")

    
    def train(self):
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Training configuration: {self.config}")
        
        try:
            for epoch in range(self.start_epoch, self.config['epochs']):
                logger.info(f"Epoch: {epoch + 1}/{self.config['epochs']}")
                
                try:
                    start_time = time.time()
                    # Start training and validation
                    train_loss, train_acc = self.train_epoch()
                    val_start_time = time.time()
                    val_loss, val_acc = self.validate()
                    val_time = time.time() - val_start_time
                    epoch_time = time.time() - start_time
                    
                    # Add the epoch summary log here
                    logger.info(f"Epoch {epoch + 1} Summary: Train Loss = {train_loss:.4f}, "
                              f"Train Acc = {train_acc:.2f}%, Val Loss = {val_loss:.4f}, "
                              f"Val Acc = {val_acc:.2f}%, Epoch Time = {epoch_time:.2f}s, "
                              f"Validation Time = {val_time:.2f}s")
                    
                    # Store metrics and handle checkpoints
                    self._update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
                    
                except KeyboardInterrupt:
                    logger.warning("\nTraining interrupted by user during epoch %d", epoch + 1)
                    # Save checkpoint before exiting
                    self.save_checkpoint(epoch)
                    return  # Exit training loop
                
                if self.early_stopping_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Ensure cleanup happens
            try:
                self.shutdown_worker()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    def _update_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Helper method to update metrics and handle checkpoints"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # Update scheduler with validation accuracy
        self.scheduler.step(val_acc)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Save checkpoint if required
        if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
            self.save_checkpoint(epoch)
        
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.early_stopping_counter = 0
            self.save_checkpoint(epoch, is_best=True)
        else:
            self.early_stopping_counter += 1





if __name__ == '__main__':
    config = {
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 30,
        'num_workers': 4,
        'checkpoint_frequency': 5,
        # Optimizer parameters
        'weight_decay': 1e-4,
        # Additional parameters
        'warmup_epochs': 5,
        'grad_clip_value': 1.0,
        'early_stopping_patience': 10,
        'early_stopping_delta': 0.001
    }

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    trainer = None
    
    try:
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Starting cleanup...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Exception details:")  # This will print the full traceback
    finally:
        if trainer is not None:
            try:
                trainer.shutdown_worker()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")
                logger.exception("Cleanup exception details:")
