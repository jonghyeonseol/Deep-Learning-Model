import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from collections import defaultdict
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 device=None, save_dir='./checkpoints'):
        """
        Neural network trainer with comprehensive training utilities.

        Args:
            model (nn.Module): The neural network model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            test_loader (DataLoader, optional): Test data loader
            device (torch.device, optional): Device to run training on
            save_dir (str): Directory to save checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = save_dir

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model.to(self.device)

        # Training configuration
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training history
        self.history = defaultdict(list)
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def configure_optimizer(self, optimizer_name='adam', lr=0.001, weight_decay=1e-4, **kwargs):
        """
        Configure the optimizer.

        Args:
            optimizer_name (str): Name of optimizer ('adam', 'sgd', 'rmsprop')
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            **kwargs: Additional optimizer parameters
        """
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay, **kwargs)
        elif optimizer_name.lower() == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                     momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr,
                                         weight_decay=weight_decay, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        print(f"Configured {optimizer_name} optimizer with lr={lr}")

    def configure_scheduler(self, scheduler_name='step', **kwargs):
        """
        Configure learning rate scheduler.

        Args:
            scheduler_name (str): Name of scheduler ('step', 'cosine', 'plateau')
            **kwargs: Scheduler-specific parameters
        """
        if self.optimizer is None:
            raise ValueError("Configure optimizer before scheduler")

        if scheduler_name.lower() == 'step':
            step_size = kwargs.get('step_size', 10)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'cosine':
            T_max = kwargs.get('T_max', 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name.lower() == 'plateau':
            patience = kwargs.get('patience', 5)
            factor = kwargs.get('factor', 0.1)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=patience, factor=factor, verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        print(f"Configured {scheduler_name} scheduler")

    def configure_criterion(self, criterion_name='crossentropy'):
        """
        Configure loss criterion.

        Args:
            criterion_name (str): Name of criterion ('crossentropy', 'mse')
        """
        if criterion_name.lower() == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name.lower() == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

        print(f"Configured {criterion_name} loss criterion")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log progress
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch."""
        if self.val_loader is None:
            return None, None

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def test(self):
        """Test the model on test set."""
        if self.test_loader is None:
            print("No test loader provided")
            return None, None

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        return test_loss, test_acc

    def train(self, epochs, early_stopping_patience=None, save_best=True):
        """
        Train the model for specified number of epochs.

        Args:
            epochs (int): Number of epochs to train
            early_stopping_patience (int, optional): Early stopping patience
            save_best (bool): Whether to save the best model
        """
        if self.optimizer is None:
            self.configure_optimizer()
        if self.criterion is None:
            self.configure_criterion()

        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()

        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            print(f'\nEpoch {self.current_epoch}/{epochs}')
            print('-' * 50)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate_epoch()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, self.current_epoch)
            if val_loss is not None:
                self.writer.add_scalar('Loss/Val', val_loss, self.current_epoch)
                self.writer.add_scalar('Accuracy/Val', val_acc, self.current_epoch)

            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            if val_loss is not None:
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Save best model
            if save_best and val_loss is not None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f'best_model.pth')
                    print(f'New best model saved! Val Acc: {val_acc:.2f}%')
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {self.current_epoch} epochs')
                break

        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.2f} seconds')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')

        # Close tensorboard writer
        self.writer.close()

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': dict(self.history),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.history = defaultdict(list, checkpoint['history'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f'Checkpoint loaded: {filepath}')

    def get_learning_rate(self):
        """Get current learning rate."""
        if self.optimizer is None:
            return None
        return self.optimizer.param_groups[0]['lr']