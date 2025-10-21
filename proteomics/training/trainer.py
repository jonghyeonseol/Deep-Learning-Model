"""
Trainer Module

Independent training loop for proteomics models.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteomicsTrainer:
    """
    Trainer for proteomics deep learning models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'auto',
        checkpoint_dir: str = './checkpoints/proteomics',
        patience: int = 10,
        use_amp: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device ('cuda', 'cpu', 'mps', or 'auto')
            checkpoint_dir: Directory for saving checkpoints
            patience: Early stopping patience
            use_amp: Use automatic mixed precision
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.patience = patience
        self.use_amp = use_amp

        # Setup device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # AMP scaler
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            self.use_amp = False

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Skip batches with None labels
            if target is None or target[0] is None:
                continue

            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * data.size(0)
            total += target.size(0)

            # Calculate accuracy (for classification)
            if output.shape[1] > 1:  # Multi-class classification
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                if target is None or target[0] is None:
                    continue

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                total += target.size(0)

                if output.shape[1] > 1:
                    _, predicted = output.max(1)
                    correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(self, num_epochs: int, verbose: bool = True):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            verbose: Print progress
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_metric'].append(train_metrics['accuracy'])

            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metric'].append(val_metrics['accuracy'])

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            epoch_time = time.time() - start_time

            # Print progress
            if verbose:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                logger.info(f"✓ New best model saved (Val Loss: {val_metrics['loss']:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

        logger.info(f"Training complete! Best epoch: {self.best_epoch}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
