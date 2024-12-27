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
import pickle
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt

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
    def __init__(self, transform=None, is_train=True, split_dir='/home/ubuntu/imagenet_ec2/data/'):
        self.transform = transform
        self.is_train = is_train
        split_file = os.path.join(split_dir, 'train_indices.pkl' if is_train else 'val_indices.pkl')
        dataset_split = 'train' if is_train else 'validation'

        if not os.path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)

        # Load dataset
        logger.info(f"Loading {'training' if is_train else 'validation'} dataset...")
        full_dataset = load_dataset(
            'imagenet-1k',
            split=dataset_split,
            cache_dir='/home/ubuntu/.cache/huggingface/datasets'
        )

        if os.path.exists(split_file):
            logger.info(f"Loading saved split indices from {split_file}...")
            with open(split_file, 'rb') as f:
                indices = pickle.load(f)
            self.dataset = full_dataset.select(indices)
        else:
            logger.info("Generating new split and saving indices...")
            indices = list(range(len(full_dataset)))
            with open(split_file, 'wb') as f:
                pickle.dump(indices, f)
            self.dataset = full_dataset

        logger.info(f"Dataset loaded with {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            sample = self.dataset[idx]
            image = sample['image']  # PIL Image
            label = sample['label']

            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            if self.transform:
                image = self.transform(image)
                
            # Add debug check for tensor shape
            if isinstance(image, torch.Tensor):
                if image.shape[0] != 3:
                    logger.error(f"Incorrect number of channels: {image.shape} at index {idx}")
                    raise ValueError(f"Expected 3 channels, got {image.shape[0]}")
                
            return image, label
            
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {str(e)}")
            raise

class Trainer:
    
    def __init__(self, config):
        self.config = config
        logger.info("Initializing trainer...")
        
        # No need to check data directory since we're using HuggingFace datasets
        logger.info("Using HuggingFace datasets for ImageNet")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_acc = 0

        # 1. Setup data first
        self.setup_data()
        logger.info("Data loaders initialized")

        # 2. Initialize model, criterion, and optimizer
        self.model = ImageNetModel(num_classes=1000, pretrained=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )
        logger.info("Model, criterion, and optimizer initialized")

        # 3. Load checkpoint - This will update model, optimizer, and epoch states
        self.load_checkpoint()
        logger.info(f"Starting from epoch {self.start_epoch}")

        # 4. Initialize scheduler with remaining steps
        steps_per_epoch = len(self.train_loader)
        remaining_epochs = config['epochs'] - self.start_epoch
        total_steps = steps_per_epoch * remaining_epochs
        
        logger.info(f"Initializing OneCycleLR scheduler:")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Remaining epochs: {remaining_epochs}")
        logger.info(f"- Total remaining steps: {total_steps}")
        
        if total_steps <= 0:
            raise ValueError(f"No training steps remaining. Current epoch {self.start_epoch} >= max epochs {config['epochs']}")
        
        # Initialize OneCycleLR scheduler with remaining steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'] * 10,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        logger.info("Scheduler initialized")

        # Initialize async checkpoint saving
        self.checkpoint_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        self.checkpoint_worker = threading.Thread(
            target=self._run_checkpoint_worker, 
            daemon=True
        )
        self.checkpoint_worker.start()

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
        
        self.plots_dir = os.path.join('/home/ubuntu/imagenet_ec2', 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

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
            transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure RGB
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure RGB
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
        scaler = GradScaler('cuda')

        logger.info("Starting training epoch...")
        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader)):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast('cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            scaler.step(self.optimizer)
            scaler.update()

            # Step the scheduler after each batch
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")

        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        torch.cuda.empty_cache()  # Clear GPU cache before validation
        
        total_loss = 0
        correct = 0
        total = 0

        logger.info("Starting validation...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)  # Only get top-1 prediction
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 100 == 0:
                    logger.info(f"Validation Progress: Batch {batch_idx}/{len(self.val_loader)}")

        accuracy = 100. * correct / total
        logger.info(f"Validation completed. Avg Loss = {total_loss / len(self.val_loader):.4f}, "
                    f"Top-1 Accuracy = {accuracy:.2f}%")
        
        torch.cuda.empty_cache()
        return total_loss / len(self.val_loader), accuracy


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
            self.start_epoch = 0
            self.best_acc = 0

    
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

        # Plot progress after updating metrics
        self.plot_training_progress()

    def find_lr(self, num_iter=100):
        """Find optimal learning rate using LR Finder"""
        logger.info("Starting learning rate finder...")
        
        # Create a copy of the model for LR finding
        model_copy = ImageNetModel(num_classes=1000, pretrained=True).to(self.device)
        model_copy.load_state_dict(self.model.state_dict())
        
        optimizer = optim.Adam(
            model_copy.parameters(),
            lr=1e-7,  # Start with a very low learning rate
            weight_decay=self.config['weight_decay']
        )
        
        lr_finder = LRFinder(model_copy, optimizer, self.criterion, device=self.device)
        
        try:
            lr_finder.range_test(
                self.train_loader,
                end_lr=10,  # End with a high learning rate
                num_iter=num_iter,
                step_mode="exp",
                diverge_th=5,
            )
            
            # Plot the learning rate finder results
            fig, ax = plt.subplots(figsize=(10, 6))
            lr_finder.plot()
            plt.title('Learning Rate Finder Results')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'lr_finder.png')
            plt.savefig(plot_path)
            plt.close()
            
            # Get suggestion for learning rate
            suggested_lr = lr_finder.suggestion()
            logger.info(f"Suggested learning rate: {suggested_lr:.6f}")
            
            # Save learning rate history
            history_path = os.path.join(self.plots_dir, 'lr_finder_history.txt')
            with open(history_path, 'w') as f:
                f.write(f"Suggested LR: {suggested_lr}\n")
                f.write("History:\n")
                for lr, loss in zip(lr_finder.history['lr'], lr_finder.history['loss']):
                    f.write(f"LR: {lr:.8f}, Loss: {loss:.8f}\n")
            
            return suggested_lr
            
        except Exception as e:
            logger.error(f"Error during LR finding: {e}")
            raise
        finally:
            # Clean up
            del model_copy
            torch.cuda.empty_cache()

    def plot_training_progress(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'training_progress_epoch_{len(self.train_losses)}.png')
        plt.savefig(plot_path)
        plt.close()

    def is_fresh_training(self):
        """Check if this is a fresh training run"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'last_checkpoint.pth')
        lr_finder_path = os.path.join(self.plots_dir, 'lr_finder.png')
        return not (os.path.exists(checkpoint_path) or os.path.exists(lr_finder_path))





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
        
        # Check if this is a fresh training run
        if trainer.is_fresh_training():
            logger.info("Fresh training detected. Running learning rate finder...")
            try:
                suggested_lr = trainer.find_lr(num_iter=100)
                config['learning_rate'] = suggested_lr
                logger.info(f"Using suggested learning rate: {suggested_lr}")
                
                # Reinitialize trainer with new learning rate
                del trainer
                torch.cuda.empty_cache()
                trainer = Trainer(config)
            except Exception as e:
                logger.error(f"LR finder failed: {e}. Using default learning rate.")
        else:
            logger.info("Resuming training from checkpoint. Skipping LR finder.")
        
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Starting cleanup...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Exception details:")
    finally:
        if trainer is not None:
            try:
                trainer.shutdown_worker()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")
                logger.exception("Cleanup exception details:")
