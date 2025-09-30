import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from collections import deque, defaultdict
import threading
import time
import queue
import warnings
warnings.filterwarnings('ignore')

# Enable interactive plotting
plt.ion()


class LiveTrainingMonitor:
    """Real-time training monitor with live updating plots."""

    def __init__(self, update_frequency=10, max_points=1000):
        """
        Initialize real-time training monitor.

        Args:
            update_frequency (int): Update plots every N batches
            max_points (int): Maximum points to keep in memory
        """
        self.update_frequency = update_frequency
        self.max_points = max_points

        # Data storage
        self.data = {
            'batch': deque(maxlen=max_points),
            'epoch': deque(maxlen=max_points),
            'train_loss': deque(maxlen=max_points),
            'train_acc': deque(maxlen=max_points),
            'val_loss': deque(maxlen=max_points),
            'val_acc': deque(maxlen=max_points),
            'learning_rate': deque(maxlen=max_points),
            'gradient_norm': deque(maxlen=max_points)
        }

        # Setup live plots
        self.fig = None
        self.axes = None
        self.lines = {}
        self.setup_plots()

        # Control variables
        self.is_monitoring = False
        self.batch_count = 0

    def setup_plots(self):
        """Setup the live plotting interface."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('🧠 Real-time ANN Training Monitor', fontsize=16, fontweight='bold')

        # Loss plot
        ax = self.axes[0, 0]
        self.lines['train_loss'], = ax.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.lines['val_loss'], = ax.plot([], [], 'r-', label='Val Loss', linewidth=2)
        ax.set_title('📉 Loss Evolution')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy plot
        ax = self.axes[0, 1]
        self.lines['train_acc'], = ax.plot([], [], 'g-', label='Train Acc', linewidth=2)
        self.lines['val_acc'], = ax.plot([], [], 'orange', label='Val Acc', linewidth=2)
        ax.set_title('📈 Accuracy Evolution')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate plot
        ax = self.axes[0, 2]
        self.lines['learning_rate'], = ax.plot([], [], 'm-', linewidth=2)
        ax.set_title('⚡ Learning Rate')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Gradient norm plot
        ax = self.axes[1, 0]
        self.lines['gradient_norm'], = ax.plot([], [], 'c-', linewidth=2)
        ax.set_title('🌊 Gradient Flow')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Loss distribution
        ax = self.axes[1, 1]
        ax.set_title('📊 Loss Distribution')
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')

        # Training speed
        ax = self.axes[1, 2]
        ax.set_title('🚀 Training Progress')
        ax.set_xlabel('Time')
        ax.set_ylabel('Epoch Progress')

        plt.tight_layout()
        plt.show(block=False)
        plt.draw()

    def start_monitoring(self):
        """Start real-time monitoring."""
        self.is_monitoring = True
        self.batch_count = 0
        print("🔴 Real-time monitoring STARTED")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        print("⏹️ Real-time monitoring STOPPED")

    def update(self, epoch, batch, train_loss, train_acc=None, val_loss=None,
               val_acc=None, learning_rate=None, gradient_norm=None):
        """Update monitoring data and plots."""
        if not self.is_monitoring:
            return

        self.batch_count += 1

        # Store data
        self.data['batch'].append(self.batch_count)
        self.data['epoch'].append(epoch)
        self.data['train_loss'].append(train_loss)
        self.data['train_acc'].append(train_acc or 0)
        self.data['val_loss'].append(val_loss or 0)
        self.data['val_acc'].append(val_acc or 0)
        self.data['learning_rate'].append(learning_rate or 0)
        self.data['gradient_norm'].append(gradient_norm or 0)

        # Update plots every N batches
        if self.batch_count % self.update_frequency == 0:
            self._update_plots()

    def _update_plots(self):
        """Update all live plots."""
        try:
            batches = list(self.data['batch'])

            if len(batches) < 2:
                return

            # Update loss plot
            self.lines['train_loss'].set_data(batches, list(self.data['train_loss']))
            if any(self.data['val_loss']):
                self.lines['val_loss'].set_data(batches, list(self.data['val_loss']))
            self._rescale_axis(self.axes[0, 0])

            # Update accuracy plot
            self.lines['train_acc'].set_data(batches, list(self.data['train_acc']))
            if any(self.data['val_acc']):
                self.lines['val_acc'].set_data(batches, list(self.data['val_acc']))
            self._rescale_axis(self.axes[0, 1])

            # Update learning rate
            if any(self.data['learning_rate']):
                lr_data = [lr for lr in self.data['learning_rate'] if lr > 0]
                if lr_data:
                    self.lines['learning_rate'].set_data(batches[-len(lr_data):], lr_data)
                    self._rescale_axis(self.axes[0, 2])

            # Update gradient norm
            if any(self.data['gradient_norm']):
                grad_data = [gn for gn in self.data['gradient_norm'] if gn > 0]
                if grad_data:
                    self.lines['gradient_norm'].set_data(batches[-len(grad_data):], grad_data)
                    self._rescale_axis(self.axes[1, 0])

            # Update loss distribution
            self.axes[1, 1].clear()
            if len(self.data['train_loss']) > 10:
                self.axes[1, 1].hist(list(self.data['train_loss']), bins=20, alpha=0.7, color='blue')
                self.axes[1, 1].set_title('📊 Loss Distribution')
                self.axes[1, 1].set_xlabel('Loss Value')
                self.axes[1, 1].set_ylabel('Frequency')

            # Update training progress
            self.axes[1, 2].clear()
            epochs = list(self.data['epoch'])
            if epochs:
                current_epoch = epochs[-1]
                progress = (self.batch_count % 100) / 100  # Approximate progress
                self.axes[1, 2].barh(0, progress, color='green', alpha=0.7)
                self.axes[1, 2].set_xlim(0, 1)
                self.axes[1, 2].set_title(f'🚀 Epoch {current_epoch} Progress')
                self.axes[1, 2].set_xlabel('Progress')

            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

        except Exception as e:
            print(f"Plot update error: {e}")

    def _rescale_axis(self, ax):
        """Automatically rescale axis to fit data."""
        try:
            ax.relim()
            ax.autoscale_view()
        except:
            pass


class LiveLayerMonitor:
    """Monitor layer activations and weights in real-time."""

    def __init__(self, model, update_frequency=50):
        """
        Initialize live layer monitor.

        Args:
            model: PyTorch model to monitor
            update_frequency (int): Update every N batches
        """
        self.model = model
        self.update_frequency = update_frequency
        self.hooks = []
        self.activations = {}
        self.weights = {}
        self.batch_count = 0

        # Setup monitoring
        self.fig = None
        self.is_monitoring = False
        self.setup_hooks()

    def setup_hooks(self):
        """Setup hooks to capture layer data."""
        def activation_hook(name):
            def hook(module, input, output):
                if self.is_monitoring and isinstance(output, torch.Tensor):
                    # Store activation statistics
                    with torch.no_grad():
                        data = output.detach().cpu().numpy()
                        self.activations[name] = {
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'min': float(np.min(data)),
                            'max': float(np.max(data)),
                            'sparsity': float(np.mean(data == 0))  # For ReLU-like activations
                        }
            return hook

        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.hooks.append(module.register_forward_hook(activation_hook(name)))
                # Store initial weights
                self.weights[name] = {
                    'mean': float(module.weight.mean().item()),
                    'std': float(module.weight.std().item())
                }

    def start_monitoring(self):
        """Start live layer monitoring."""
        self.is_monitoring = True
        self.batch_count = 0
        self.setup_plots()
        print("🔴 Live layer monitoring STARTED")

    def stop_monitoring(self):
        """Stop live layer monitoring."""
        self.is_monitoring = False
        if self.fig:
            plt.close(self.fig)
        print("⏹️ Live layer monitoring STOPPED")

    def setup_plots(self):
        """Setup live layer monitoring plots."""
        n_layers = len(self.activations) if self.activations else len(self.weights)
        if n_layers == 0:
            return

        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🧠 Live Layer Activity Monitor', fontsize=16, fontweight='bold')

        plt.show(block=False)

    def update(self):
        """Update layer monitoring display."""
        if not self.is_monitoring or not self.activations:
            return

        self.batch_count += 1

        if self.batch_count % self.update_frequency == 0:
            self._update_layer_plots()

    def _update_layer_plots(self):
        """Update layer activity plots."""
        try:
            if not self.fig or not self.activations:
                return

            layer_names = list(self.activations.keys())

            # Plot 1: Activation means
            ax = self.axes[0, 0]
            ax.clear()
            means = [self.activations[name]['mean'] for name in layer_names]
            bars = ax.bar(range(len(layer_names)), means, color='skyblue', alpha=0.7)
            ax.set_title('⚡ Layer Activation Means')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Activation')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)

            # Add value labels
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

            # Plot 2: Activation standard deviations
            ax = self.axes[0, 1]
            ax.clear()
            stds = [self.activations[name]['std'] for name in layer_names]
            bars = ax.bar(range(len(layer_names)), stds, color='lightcoral', alpha=0.7)
            ax.set_title('📊 Layer Activation Std Dev')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Std Dev')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)

            # Plot 3: Sparsity (for ReLU-like activations)
            ax = self.axes[1, 0]
            ax.clear()
            sparsities = [self.activations[name]['sparsity'] * 100 for name in layer_names]
            bars = ax.bar(range(len(layer_names)), sparsities, color='lightgreen', alpha=0.7)
            ax.set_title('🕳️ Activation Sparsity (%)')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Sparsity (%)')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)

            # Plot 4: Activation ranges
            ax = self.axes[1, 1]
            ax.clear()
            mins = [self.activations[name]['min'] for name in layer_names]
            maxs = [self.activations[name]['max'] for name in layer_names]
            x_pos = range(len(layer_names))
            ax.bar(x_pos, maxs, color='red', alpha=0.5, label='Max')
            ax.bar(x_pos, mins, color='blue', alpha=0.5, label='Min')
            ax.set_title('📏 Activation Ranges')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Activation Value')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)
            ax.legend()

            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

        except Exception as e:
            print(f"Layer plot update error: {e}")

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class LiveNeuralNetworkVisualizer:
    """Real-time visualization of neural network thinking process."""

    def __init__(self, model):
        """
        Initialize live neural network visualizer.

        Args:
            model: PyTorch model to visualize
        """
        self.model = model
        self.fig = None
        self.is_visualizing = False

    def start_visualization(self):
        """Start live neural network visualization."""
        self.is_visualizing = True
        self.setup_visualization()
        print("🎨 Live neural network visualization STARTED")

    def stop_visualization(self):
        """Stop live neural network visualization."""
        self.is_visualizing = False
        if self.fig:
            plt.close(self.fig)
        print("🎨 Live neural network visualization STOPPED")

    def setup_visualization(self):
        """Setup the neural network visualization."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 10))
        self.ax.set_title('🧠 Live Neural Network Thinking Process', fontsize=16, fontweight='bold')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.axis('off')

        # Draw network structure
        self._draw_network_structure()

        plt.show(block=False)

    def _draw_network_structure(self):
        """Draw the basic network structure."""
        # This is a simplified representation
        # Input layer
        for i in range(3):
            circle = plt.Circle((1, 2 + i * 1.5), 0.3, color='lightblue', ec='black')
            self.ax.add_patch(circle)
            self.ax.text(1, 2 + i * 1.5, f'I{i+1}', ha='center', va='center', fontweight='bold')

        # Hidden layers
        for layer in range(2):
            for i in range(4):
                circle = plt.Circle((3 + layer * 2, 1.5 + i * 1.2), 0.25,
                                  color='lightcoral', ec='black')
                self.ax.add_patch(circle)
                self.ax.text(3 + layer * 2, 1.5 + i * 1.2, f'H{layer+1}.{i+1}',
                           ha='center', va='center', fontsize=8, fontweight='bold')

        # Output layer
        for i in range(2):
            circle = plt.Circle((8, 3 + i * 1), 0.3, color='lightgreen', ec='black')
            self.ax.add_patch(circle)
            self.ax.text(8, 3 + i * 1, f'O{i+1}', ha='center', va='center', fontweight='bold')

        self.fig.canvas.draw()

    def update_visualization(self, activations):
        """Update the visualization with current activations."""
        if not self.is_visualizing:
            return

        # This would update neuron colors based on activation levels
        # Implementation would depend on specific requirements
        pass


def create_comprehensive_monitor(model):
    """Create a comprehensive real-time monitoring system."""

    class ComprehensiveMonitor:
        def __init__(self, model):
            self.training_monitor = LiveTrainingMonitor()
            self.layer_monitor = LiveLayerMonitor(model)
            self.network_visualizer = LiveNeuralNetworkVisualizer(model)

        def start_all(self):
            """Start all monitoring components."""
            print("🚀 Starting comprehensive real-time monitoring...")
            self.training_monitor.start_monitoring()
            self.layer_monitor.start_monitoring()
            self.network_visualizer.start_visualization()
            print("✅ All monitors active!")

        def stop_all(self):
            """Stop all monitoring components."""
            print("⏹️ Stopping all monitors...")
            self.training_monitor.stop_monitoring()
            self.layer_monitor.stop_monitoring()
            self.network_visualizer.stop_visualization()
            print("✅ All monitors stopped!")

        def update_training(self, epoch, batch, train_loss, train_acc=None,
                          val_loss=None, val_acc=None, learning_rate=None,
                          gradient_norm=None):
            """Update training monitoring."""
            self.training_monitor.update(epoch, batch, train_loss, train_acc,
                                       val_loss, val_acc, learning_rate, gradient_norm)

        def update_layers(self):
            """Update layer monitoring."""
            self.layer_monitor.update()

        def cleanup(self):
            """Clean up resources."""
            self.layer_monitor.clear_hooks()

    return ComprehensiveMonitor(model)