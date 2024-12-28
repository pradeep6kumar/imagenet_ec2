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
from datasets import load_dataset
import numpy as np

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

class ImageNetDataset(Dataset):
    def __init__(self, transform=None, is_train=True):
        self.transform = transform
        
        # Load the ImageNet dataset from HuggingFace
        logger.info("Loading ImageNet dataset from HuggingFace...")
        self.dataset = load_dataset(
            'imagenet-1k',
            split='train' if is_train else 'validation',
            cache_dir='/home/ubuntu/.cache/huggingface/datasets'
        )
        
        logger.info(f"Dataset loaded with {len(self.dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get sample from the HuggingFace dataset
        sample = self.dataset[idx]
        
        # Convert image from PIL to RGB if it's not already
        image = sample['image']  # PIL Image
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Add this line to ensure RGB
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class Trainer:
    
    def __init__(self, config):
        self.config = config
        logger.info("Initializing trainer...")
        
        # No need to check data directory since we're using HuggingFace datasets
        logger.info("Using HuggingFace datasets for ImageNet")
        
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

        # Initialize model with 1000 classes for full ImageNet
        self.model = ImageNetModel(num_classes=1000, pretrained=True).to(self.device)

        # Loss and optimizer with simple parameters
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer (SGD instead of Adam)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,  # Important for SGD
            weight_decay=config['weight_decay'],
            nesterov=True  # Use Nesterov momentum
        )

        # Setup data first to get loader length
        self.setup_data()

        # Replace the ReduceLROnPlateau scheduler with MultiStepLR
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['lr_schedule']['milestones'],
            gamma=config['lr_schedule']['gamma']
        )

        # Add warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=len(self.train_loader) * config['warmup_epochs']
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
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandAugment(num_ops=2, magnitude=9),
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

        # Use the new ImageNetDataset class
        self.train_dataset = ImageNetDataset(
            transform=train_transform,
            is_train=True
        )
        
        self.val_dataset = ImageNetDataset(
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
        
        # Initialize GradScaler for mixed precision training
        scaler = GradScaler(
            init_scale=2**16,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True
        )
        
        # Improved progress tracking
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        batch_start_time = time.time()
        
        logger.info("Starting training epoch with mixed precision...")
        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader)):
            images = images.to(self.device, non_blocking=True)  # Added non_blocking=True
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with autocast(device_type='cuda', dtype=torch.float16):  # Explicitly set dtype
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            # Step with scaler
            scaler.step(self.optimizer)
            scaler.update()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update running metrics
            running_loss += loss.item()
            running_correct += predicted.eq(labels).sum().item()
            running_total += labels.size(0)

            if batch_idx % 100 == 0:
                # Calculate speed
                batch_time = time.time() - batch_start_time
                images_per_sec = (self.config['batch_size'] * 100) / batch_time if batch_idx > 0 else 0
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log detailed batch information
                logger.info(f"\nBatch {batch_idx}/{len(self.train_loader)}:")
                logger.info(f"  - Loss: {running_loss/100:.3f}")
                logger.info(f"  - Accuracy: {100.*running_correct/running_total:.2f}%")
                logger.info(f"  - Speed: {images_per_sec:.1f} img/s")
                logger.info(f"  - LR: {current_lr:.6f}")
                
                # Reset running metrics
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                batch_start_time = time.time()
                
                # Log GPU memory usage
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    gpu_utilization = torch.cuda.utilization()
                    logger.info(f"GPU: {gpu_utilization}% | Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f}MB | CUDA Allocated: {gpu_memory_used:.0f}MB")

            # Clear cache periodically
            if batch_idx % 500 == 0:
                torch.cuda.empty_cache()

        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        torch.cuda.empty_cache()
        
        total_loss = 0
        correct = 0
        total = 0
        top5_correct = 0

        logger.info("Starting validation with mixed precision...")
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader)):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.topk(5, dim=1)
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
                    
                    # Update learning rate schedulers
                    if epoch < self.config['warmup_epochs']:
                        self.warmup_scheduler.step()
                    else:
                        self.scheduler.step()
                    
                    # Store metrics and handle checkpoints
                    self._update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
                    
                except KeyboardInterrupt:
                    logger.warning("\nTraining interrupted by user during epoch %d", epoch + 1)
                    self.save_checkpoint(epoch)
                    return
                
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
        
        # Remove the val_acc parameter
        self.scheduler.step()  # Just call step() without parameters
        
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
        'checkpoint_dir': '/home/ubuntu/imagenet_ec2/checkpoints',
        'batch_size': 256,
        'learning_rate': 0.1,
        'epochs': 100,
        'num_workers': 8,
        'checkpoint_frequency': 5,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        'grad_clip_value': 1.0,
        'early_stopping_patience': 15,
        'early_stopping_delta': 0.001,
        'lr_schedule': {
            'milestones': [30, 60, 90],
            'gamma': 0.1
        },
        'prefetch_factor': 4,
        'pin_memory': True,
        'gradient_accumulation_steps': 4
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
