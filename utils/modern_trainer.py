"""
Modern training utilities with state-of-the-art techniques.

This module implements:
- AdamW optimizer (decoupled weight decay)
- Cosine annealing with warmup
- Mixed precision training (AMP)
- Gradient clipping and accumulation
- Label smoothing
- EMA (Exponential Moving Average)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import time
import os
from collections import defaultdict
import copy


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.

    Prevents the model from becoming over-confident by distributing
    some probability mass to incorrect classes.

    Instead of targets [0, 0, 1, 0] (one-hot),
    use smoothed targets [0.03, 0.03, 0.91, 0.03] (with smoothing=0.1)

    Args:
        smoothing: Label smoothing factor (0.0 to 1.0, typically 0.1)
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Restarts and Linear Warmup.

    Combines:
    1. Linear warmup: Gradually increases LR from 0 to max_lr
    2. Cosine annealing: Smoothly decreases LR following cosine curve
    3. Warm restarts: Periodically resets LR to restart training

    Args:
        optimizer: Wrapped optimizer
        first_cycle_steps: Number of steps in first cycle
        warmup_steps: Number of warmup steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        gamma: Decay factor for cycle length after restart
    """

    def __init__(self, optimizer, first_cycle_steps=200, warmup_steps=50,
                 max_lr=0.001, min_lr=0.00001, gamma=1.0):
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma

        self.current_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_count = 0

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, -1)

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.current_cycle_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265359)))

        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        self.step_count += 1

        if self.step_count >= self.current_cycle_steps:
            # Restart
            self.cycle += 1
            self.step_count = 0
            self.current_cycle_steps = int(self.current_cycle_steps * self.gamma)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains a moving average of model weights for better generalization.
    Often improves test performance by 0.2-0.5%.

    Args:
        model: The model to track
        decay: Decay rate (typically 0.999 or 0.9999)
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        """Replace model parameters with EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class ModernTrainer:
    """
    Modern trainer with state-of-the-art training techniques.

    Features:
    - AdamW optimizer with decoupled weight decay
    - Cosine annealing with warmup
    - Automatic Mixed Precision (AMP) for 2-3x speedup
    - Gradient clipping for stability
    - Gradient accumulation for large batch simulation
    - Label smoothing for better calibration
    - EMA for improved generalization
    - TensorBoard logging

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        save_dir: Directory for checkpoints
    """

    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 device=None, save_dir='./checkpoints'):
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
        self.scaler = None
        self.ema = None

        # Training settings
        self.use_amp = False
        self.gradient_clip = None
        self.accumulation_steps = 1

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

    def configure_optimizer(self, optimizer_name='adamw', lr=0.001, weight_decay=0.01, **kwargs):
        """
        Configure optimizer.

        Modern best practice: Use AdamW with weight_decay=0.01 to 0.1

        Args:
            optimizer_name: 'adamw', 'adam', 'sgd'
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
        """
        if optimizer_name.lower() == 'adamw':
            # AdamW: Adam with decoupled weight decay (better than Adam)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999))
            )
        elif optimizer_name.lower() == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            nesterov = kwargs.get('nesterov', True)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        print(f"✓ Configured {optimizer_name} optimizer (lr={lr}, weight_decay={weight_decay})")

    def configure_scheduler(self, scheduler_name='cosine_warmup', epochs=100, **kwargs):
        """
        Configure learning rate scheduler.

        Modern best practice: Use cosine annealing with warmup

        Args:
            scheduler_name: 'cosine_warmup', 'cosine', 'step', 'plateau'
            epochs: Total training epochs
        """
        if self.optimizer is None:
            raise ValueError("Configure optimizer before scheduler")

        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch

        if scheduler_name.lower() == 'cosine_warmup':
            warmup_steps = kwargs.get('warmup_steps', steps_per_epoch * 5)  # 5 epochs warmup
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=total_steps,
                warmup_steps=warmup_steps,
                max_lr=self.optimizer.param_groups[0]['lr'],
                min_lr=kwargs.get('min_lr', 1e-6)
            )
            print(f"✓ Configured cosine annealing with {warmup_steps} warmup steps")

        elif scheduler_name.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=kwargs.get('min_lr', 1e-6)
            )
            print(f"✓ Configured cosine annealing scheduler")

        elif scheduler_name.lower() == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            print(f"✓ Configured step LR scheduler")

        elif scheduler_name.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=kwargs.get('patience', 10),
                factor=kwargs.get('factor', 0.1)
            )
            print(f"✓ Configured plateau LR scheduler")

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def configure_criterion(self, criterion_name='crossentropy', label_smoothing=0.0):
        """
        Configure loss criterion.

        Modern best practice: Use label smoothing (0.1) for better calibration

        Args:
            criterion_name: 'crossentropy', 'label_smoothing'
            label_smoothing: Smoothing factor (0.0 to 1.0, typically 0.1)
        """
        if criterion_name.lower() == 'label_smoothing' or label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            print(f"✓ Configured label smoothing cross entropy (smoothing={label_smoothing})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"✓ Configured cross entropy loss")

    def enable_amp(self):
        """
        Enable Automatic Mixed Precision training.

        Uses float16 for faster training (2-3x speedup on modern GPUs).
        Only works on CUDA devices with Tensor Cores (V100, RTX series, A100, etc.)
        """
        if self.device.type == 'cuda':
            self.use_amp = True
            self.scaler = GradScaler()
            print(f"✓ Enabled mixed precision training (AMP)")
        else:
            print("⚠ AMP requires CUDA device, skipping")

    def enable_gradient_clipping(self, max_norm=1.0):
        """
        Enable gradient clipping to prevent exploding gradients.

        Args:
            max_norm: Maximum gradient norm (typically 0.5 to 5.0)
        """
        self.gradient_clip = max_norm
        print(f"✓ Enabled gradient clipping (max_norm={max_norm})")

    def enable_gradient_accumulation(self, steps=2):
        """
        Enable gradient accumulation to simulate larger batch sizes.

        Effective batch size = batch_size * accumulation_steps

        Args:
            steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = steps
        print(f"✓ Enabled gradient accumulation ({steps} steps)")
        print(f"  Effective batch size: {self.train_loader.batch_size * steps}")

    def enable_ema(self, decay=0.999):
        """
        Enable Exponential Moving Average of model weights.

        Args:
            decay: EMA decay rate (typically 0.999 or 0.9999)
        """
        self.ema = EMA(self.model, decay=decay)
        print(f"✓ Enabled EMA (decay={decay})")

    def train_epoch(self):
        """Train for one epoch with modern techniques."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.accumulation_steps
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update EMA
                if self.ema is not None:
                    self.ema.update()

                # Update scheduler (per-step for warmup)
                if self.scheduler is not None and isinstance(self.scheduler, CosineAnnealingWarmupRestarts):
                    self.scheduler.step()

            # Statistics
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log progress
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item() * self.accumulation_steps:.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, '
                      f'LR: {current_lr:.6f}')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, use_ema=False):
        """Validate for one epoch."""
        if self.val_loader is None:
            return None, None

        # Apply EMA weights if requested
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        val_loss += self.criterion(output, target).item()
                else:
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        # Restore original weights if EMA was used
        if use_ema and self.ema is not None:
            self.ema.restore()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, epochs, early_stopping_patience=None, save_best=True, use_ema_for_eval=True):
        """
        Train the model with modern techniques.

        Args:
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            save_best: Save best model
            use_ema_for_eval: Use EMA weights for validation/testing
        """
        if self.optimizer is None:
            self.configure_optimizer()
        if self.criterion is None:
            self.configure_criterion()
        if self.scheduler is None:
            self.configure_scheduler(epochs=epochs)

        print(f"\n{'='*60}")
        print(f"Starting modern training for {epochs} epochs")
        print(f"{'='*60}\n")

        start_time = time.time()
        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            print(f'\n{"="*60}')
            print(f'Epoch {self.current_epoch}/{epochs}')
            print(f'{"="*60}')

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate_epoch(use_ema=use_ema_for_eval and self.ema is not None)

            # Update scheduler (per-epoch for some schedulers)
            if self.scheduler is not None and not isinstance(self.scheduler, CosineAnnealingWarmupRestarts):
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

            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, self.current_epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)

            if val_loss is not None:
                self.writer.add_scalar('Loss/Val', val_loss, self.current_epoch)
                self.writer.add_scalar('Accuracy/Val', val_acc, self.current_epoch)

            # Print results
            print(f'\nResults:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            if val_loss is not None:
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if save_best and val_loss is not None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                    print(f'\n✓ New best model! Val Acc: {val_acc:.2f}%')
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f'\n✗ Early stopping triggered after {self.current_epoch} epochs')
                break

        total_time = time.time() - start_time
        print(f'\n{"="*60}')
        print(f'Training completed in {total_time:.2f} seconds')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        print(f'{"="*60}\n')

        self.writer.close()

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'ema_shadow': self.ema.shadow if self.ema else None,
            'history': dict(self.history),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.ema and checkpoint['ema_shadow']:
            self.ema.shadow = checkpoint['ema_shadow']

        self.current_epoch = checkpoint['epoch']
        self.history = defaultdict(list, checkpoint['history'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f'✓ Checkpoint loaded: {filepath}')

    def test(self, use_ema=True):
        """Test the model."""
        if self.test_loader is None:
            print("No test loader provided")
            return None, None

        # Apply EMA weights if requested
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        test_loss += self.criterion(output, target).item()
                else:
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        # Restore original weights if EMA was used
        if use_ema and self.ema is not None:
            self.ema.restore()

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        return test_loss, test_acc
