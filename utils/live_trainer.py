import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from collections import defaultdict
import numpy as np
from .trainer import Trainer  # Import base trainer
from .realtime_monitor import create_comprehensive_monitor


class LiveTrainer(Trainer):
    """
    Enhanced trainer with real-time monitoring capabilities.
    Shows live training progress, layer activations, and network behavior.
    """

    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 device=None, save_dir='./checkpoints', enable_live_monitoring=True):
        """
        Initialize live trainer with real-time monitoring.

        Args:
            model (nn.Module): The neural network model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            test_loader (DataLoader, optional): Test data loader
            device (torch.device, optional): Device to run training on
            save_dir (str): Directory to save checkpoints and logs
            enable_live_monitoring (bool): Enable real-time monitoring
        """
        # Initialize base trainer
        super().__init__(model, train_loader, val_loader, test_loader, device, save_dir)

        # Real-time monitoring setup
        self.enable_live_monitoring = enable_live_monitoring
        self.live_monitor = None
        self.batch_times = []
        self.gradient_norms = []

        if enable_live_monitoring:
            self.live_monitor = create_comprehensive_monitor(model)
            print("🔴 Live monitoring enabled - training will show real-time visualizations!")

    def train_epoch(self):
        """Train for one epoch with live monitoring."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start_time = time.time()

            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Calculate gradient norm for monitoring
            gradient_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    gradient_norm += param.grad.data.norm(2).item() ** 2
            gradient_norm = gradient_norm ** 0.5
            self.gradient_norms.append(gradient_norm)

            # Optimizer step
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Calculate current accuracy
            current_acc = 100. * correct / total

            # Real-time monitoring updates
            if self.enable_live_monitoring and self.live_monitor:
                # Update training monitor
                current_lr = self.get_learning_rate()
                self.live_monitor.update_training(
                    epoch=self.current_epoch,
                    batch=batch_idx + 1,
                    train_loss=loss.item(),
                    train_acc=current_acc,
                    learning_rate=current_lr,
                    gradient_norm=gradient_norm
                )

                # Update layer monitor
                self.live_monitor.update_layers()

            # Track batch time
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)

            # Log progress with live stats
            if batch_idx % 50 == 0:  # More frequent updates for live monitoring
                avg_batch_time = np.mean(self.batch_times[-10:]) if self.batch_times else 0
                print(f'⚡ Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {current_acc:.2f}%, '
                      f'Grad: {gradient_norm:.4f}, '
                      f'Time: {avg_batch_time:.3f}s/batch')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        epoch_time = time.time() - epoch_start_time
        print(f"📊 Epoch completed in {epoch_time:.2f}s")

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch with live monitoring."""
        if self.val_loader is None:
            return None, None

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Update live monitoring for validation
                if self.enable_live_monitoring and self.live_monitor and batch_idx % 20 == 0:
                    current_val_acc = 100. * correct / total
                    current_val_loss = val_loss / (batch_idx + 1)
                    self.live_monitor.update_training(
                        epoch=self.current_epoch,
                        batch=batch_idx + 1,
                        train_loss=0,  # No training in validation
                        val_loss=current_val_loss,
                        val_acc=current_val_acc
                    )

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, epochs, early_stopping_patience=None, save_best=True):
        """
        Train the model with live monitoring.

        Args:
            epochs (int): Number of epochs to train
            early_stopping_patience (int, optional): Early stopping patience
            save_best (bool): Whether to save the best model
        """
        if self.optimizer is None:
            self.configure_optimizer()
        if self.criterion is None:
            self.configure_criterion()

        # Start live monitoring
        if self.enable_live_monitoring and self.live_monitor:
            self.live_monitor.start_all()
            print("🎬 Live monitoring started - watch your ANN learn in real-time!")
            time.sleep(2)  # Give time for plots to appear

        print(f"🚀 Starting LIVE training for {epochs} epochs...")
        print("📺 Watch the real-time plots to see your neural network thinking!")
        start_time = time.time()

        patience_counter = 0

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                print(f'\n🧠 Epoch {self.current_epoch}/{epochs} - Neural Network Learning...')
                print('-' * 70)

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

                # Live monitoring final update for epoch
                if self.enable_live_monitoring and self.live_monitor:
                    current_lr = self.get_learning_rate()
                    avg_grad_norm = np.mean(self.gradient_norms[-10:]) if self.gradient_norms else 0
                    self.live_monitor.update_training(
                        epoch=self.current_epoch,
                        batch=len(self.train_loader),
                        train_loss=train_loss,
                        train_acc=train_acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        learning_rate=current_lr,
                        gradient_norm=avg_grad_norm
                    )

                # Print epoch results with visual indicators
                print(f'📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                if val_loss is not None:
                    print(f'📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

                # Performance indicators
                if val_acc is not None:
                    if val_acc > 80:
                        print("🎉 Excellent performance!")
                    elif val_acc > 60:
                        print("👍 Good progress!")
                    elif val_acc > 40:
                        print("📚 Learning steadily...")
                    else:
                        print("🤔 Still learning...")

                # Save best model
                if save_best and val_loss is not None:
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_val_loss = val_loss
                        self.save_checkpoint(f'best_model.pth')
                        print(f'💾 New best model saved! Val Acc: {val_acc:.2f}%')
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # Early stopping
                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    print(f'⏰ Early stopping triggered after {self.current_epoch} epochs')
                    break

                # Gradient health check
                if len(self.gradient_norms) > 10:
                    recent_grad_norm = np.mean(self.gradient_norms[-10:])
                    if recent_grad_norm > 10:
                        print("⚠️  Warning: Large gradients detected - might need gradient clipping")
                    elif recent_grad_norm < 1e-6:
                        print("⚠️  Warning: Very small gradients - might have vanishing gradients")

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")

        finally:
            # Stop live monitoring
            if self.enable_live_monitoring and self.live_monitor:
                print("\n📊 Stopping live monitoring...")
                time.sleep(1)  # Let final updates complete
                self.live_monitor.stop_all()
                self.live_monitor.cleanup()

        total_time = time.time() - start_time
        print(f'\n✅ Training completed in {total_time:.2f} seconds')
        print(f'🏆 Best validation accuracy: {self.best_val_acc:.2f}%')

        # Training summary
        print(f"\n📈 Training Summary:")
        print(f"   Final train accuracy: {train_acc:.2f}%")
        if val_acc is not None:
            print(f"   Final validation accuracy: {val_acc:.2f}%")
        print(f"   Average gradient norm: {np.mean(self.gradient_norms):.4f}")
        print(f"   Average batch time: {np.mean(self.batch_times):.3f}s")

    def get_training_stats(self):
        """Get detailed training statistics."""
        return {
            'gradient_norms': self.gradient_norms,
            'batch_times': self.batch_times,
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'training_efficiency': len(self.batch_times) / sum(self.batch_times) if self.batch_times else 0
        }

    def enable_monitoring(self):
        """Enable live monitoring if it was disabled."""
        if not self.enable_live_monitoring:
            self.enable_live_monitoring = True
            self.live_monitor = create_comprehensive_monitor(self.model)
            print("🔴 Live monitoring enabled!")

    def disable_monitoring(self):
        """Disable live monitoring."""
        if self.enable_live_monitoring and self.live_monitor:
            self.live_monitor.stop_all()
            self.live_monitor.cleanup()
            self.enable_live_monitoring = False
            self.live_monitor = None
            print("⏹️ Live monitoring disabled!")