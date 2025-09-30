import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from collections import defaultdict
import os


class PerceptronVisualizer:
    """Visualizes the structure and operation of perceptrons and neural networks."""

    def __init__(self, save_dir='./visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_single_perceptron(self, weights, bias, inputs=None, output=None,
                              activation_name='linear', save_path=None):
        """
        Visualize a single perceptron with weights, bias, and data flow.

        Args:
            weights (torch.Tensor or list): Input weights
            bias (float): Bias term
            inputs (list, optional): Input values
            output (float, optional): Output value
            activation_name (str): Name of activation function
            save_path (str, optional): Path to save the visualization
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Setup positions
        n_inputs = len(weights)
        input_y_positions = np.linspace(0.2, 0.8, n_inputs)
        perceptron_pos = (0.6, 0.5)
        output_pos = (0.9, 0.5)

        # Draw input nodes
        input_nodes = []
        for i, (y_pos, weight) in enumerate(zip(input_y_positions, weights)):
            circle = Circle((0.1, y_pos), 0.03, color='lightblue', ec='black')
            ax.add_patch(circle)
            input_nodes.append((0.1, y_pos))

            # Label input
            if inputs is not None:
                ax.text(0.05, y_pos, f'x{i+1}={inputs[i]:.2f}',
                       ha='right', va='center', fontsize=10)
            else:
                ax.text(0.05, y_pos, f'x{i+1}', ha='right', va='center', fontsize=10)

        # Draw perceptron (main processing unit)
        perceptron = Circle(perceptron_pos, 0.08, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(perceptron)
        ax.text(perceptron_pos[0], perceptron_pos[1], activation_name,
               ha='center', va='center', fontsize=10, weight='bold')

        # Draw bias
        bias_pos = (0.4, 0.1)
        bias_circle = Circle(bias_pos, 0.03, color='lightyellow', ec='black')
        ax.add_patch(bias_circle)
        ax.text(bias_pos[0], bias_pos[1]-0.08, f'bias={bias:.3f}',
               ha='center', va='top', fontsize=9)

        # Draw output node
        output_circle = Circle(output_pos, 0.03, color='lightgreen', ec='black')
        ax.add_patch(output_circle)
        if output is not None:
            ax.text(output_pos[0]+0.05, output_pos[1], f'y={output:.3f}',
                   ha='left', va='center', fontsize=10)
        else:
            ax.text(output_pos[0]+0.05, output_pos[1], 'y',
                   ha='left', va='center', fontsize=10)

        # Draw connections with weights
        for i, ((x_in, y_in), weight) in enumerate(zip(input_nodes, weights)):
            # Connection line
            line_color = 'red' if weight < 0 else 'blue'
            line_width = min(abs(weight) * 3 + 0.5, 5)  # Scale line width with weight magnitude

            ax.plot([x_in + 0.03, perceptron_pos[0] - 0.08],
                   [y_in, perceptron_pos[1]],
                   color=line_color, linewidth=line_width, alpha=0.7)

            # Weight label
            mid_x = (x_in + perceptron_pos[0]) / 2
            mid_y = (y_in + perceptron_pos[1]) / 2
            ax.text(mid_x, mid_y + 0.02, f'w{i+1}={weight:.3f}',
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Bias connection
        ax.plot([bias_pos[0], perceptron_pos[0] - 0.08],
               [bias_pos[1] + 0.03, perceptron_pos[1] - 0.08],
               color='purple', linewidth=2, alpha=0.7)

        # Output connection
        ax.plot([perceptron_pos[0] + 0.08, output_pos[0] - 0.03],
               [perceptron_pos[1], output_pos[1]],
               color='green', linewidth=3, alpha=0.7)

        # Add legend
        red_line = mpatches.Patch(color='red', label='Negative weight')
        blue_line = mpatches.Patch(color='blue', label='Positive weight')
        purple_line = mpatches.Patch(color='purple', label='Bias')
        ax.legend(handles=[blue_line, red_line, purple_line], loc='upper right')

        # Mathematical equation
        if inputs is not None:
            equation = f"z = "
            for i, (w, x) in enumerate(zip(weights, inputs)):
                if i > 0:
                    equation += " + " if w >= 0 else " - "
                    equation += f"{abs(w):.3f}×{x:.2f}"
                else:
                    equation += f"{w:.3f}×{x:.2f}"
            equation += f" + {bias:.3f}"

            if output is not None:
                equation += f"\ny = {activation_name}(z) = {output:.3f}"

            ax.text(0.5, 0.05, equation, ha='center', va='bottom', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Single Perceptron with {activation_name.upper()} Activation',
                    fontsize=14, weight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Perceptron visualization saved to: {save_path}")

        plt.show()
        return fig

    def plot_network_architecture(self, model, save_path=None):
        """
        Visualize the complete neural network architecture.

        Args:
            model (nn.Module): PyTorch model to visualize
            save_path (str, optional): Path to save the visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Get layer information
        layers_info = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers_info.append({
                    'name': name,
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'weights': module.weight.detach().numpy(),
                    'bias': module.bias.detach().numpy() if module.bias is not None else None
                })
            elif isinstance(module, nn.Conv2d):
                layers_info.append({
                    'name': name,
                    'type': 'Conv2d',
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size
                })

        if not layers_info:
            print("No linear or convolutional layers found in the model.")
            return

        # Calculate positions
        n_layers = len(layers_info)
        layer_x_positions = np.linspace(0.1, 0.9, n_layers + 1)  # +1 for input

        # Draw layers
        for i, layer_info in enumerate(layers_info):
            x_pos = layer_x_positions[i + 1]

            if layer_info['type'] == 'Linear':
                n_neurons = layer_info['out_features']
                neuron_y_positions = np.linspace(0.1, 0.9, min(n_neurons, 10))  # Limit display

                # Draw neurons
                for j, y_pos in enumerate(neuron_y_positions):
                    circle = Circle((x_pos, y_pos), 0.02, color='lightblue', ec='black')
                    ax.add_patch(circle)

                # If too many neurons, show "..."
                if n_neurons > 10:
                    ax.text(x_pos, 0.05, f"... ({n_neurons} neurons)",
                           ha='center', va='center', fontsize=8)

                # Layer label
                ax.text(x_pos, 0.95, f"{layer_info['name']}\n{n_neurons} neurons",
                       ha='center', va='bottom', fontsize=10, weight='bold')

            elif layer_info['type'] == 'Conv2d':
                # Draw as rectangle for conv layers
                rect = FancyBboxPatch((x_pos-0.03, 0.4), 0.06, 0.2,
                                    boxstyle="round,pad=0.01",
                                    facecolor='lightcoral', edgecolor='black')
                ax.add_patch(rect)

                ax.text(x_pos, 0.95, f"{layer_info['name']}\n{layer_info['out_channels']} filters",
                       ha='center', va='bottom', fontsize=10, weight='bold')

        # Draw input layer
        input_x = layer_x_positions[0]
        if hasattr(model, 'input_size'):
            input_size = model.input_size
        elif layers_info and layers_info[0]['type'] == 'Linear':
            input_size = layers_info[0]['in_features']
        else:
            input_size = 3  # Default for conv networks

        input_y_positions = np.linspace(0.1, 0.9, min(input_size, 10))
        for y_pos in input_y_positions:
            circle = Circle((input_x, y_pos), 0.02, color='lightgreen', ec='black')
            ax.add_patch(circle)

        ax.text(input_x, 0.95, f"Input\n{input_size} features",
               ha='center', va='bottom', fontsize=10, weight='bold')

        # Draw connections (simplified)
        for i in range(len(layer_x_positions) - 1):
            x1, x2 = layer_x_positions[i], layer_x_positions[i + 1]
            ax.plot([x1 + 0.02, x2 - 0.02], [0.5, 0.5],
                   color='gray', alpha=0.5, linewidth=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Neural Network Architecture', fontsize=16, weight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network architecture saved to: {save_path}")

        plt.show()
        return fig


class LayerMonitor:
    """Monitor layer-by-layer operations during forward pass."""

    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.hooks = []

    def register_hooks(self, model):
        """Register forward and backward hooks to monitor layer operations."""
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu().numpy()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach().cpu().numpy()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def plot_layer_activations(self, save_path=None):
        """Plot activation distributions for each layer."""
        if not self.activations:
            print("No activations recorded. Run a forward pass first.")
            return

        n_layers = len(self.activations)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(15, 8))
        if n_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (layer_name, activations) in enumerate(self.activations.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Flatten activations for histogram
            flat_activations = activations.flatten()

            ax.hist(flat_activations, bins=50, alpha=0.7, density=True)
            ax.set_title(f'{layer_name}\nMean: {flat_activations.mean():.3f}')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(self.activations), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Layer-wise Activation Distributions', fontsize=16, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer activations saved to: {save_path}")

        plt.show()
        return fig

    def plot_gradient_flow(self, save_path=None):
        """Plot gradient magnitudes through the network."""
        if not self.gradients:
            print("No gradients recorded. Run a backward pass first.")
            return

        layer_names = list(self.gradients.keys())
        gradient_norms = []

        for layer_name in layer_names:
            grad = self.gradients[layer_name]
            norm = np.linalg.norm(grad.flatten())
            gradient_norms.append(norm)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(layer_names)), gradient_norms,
                      color='skyblue', edgecolor='black', alpha=0.7)

        plt.xlabel('Layer')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Flow Through Network Layers', fontsize=14, weight='bold')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, norm in zip(bars, gradient_norms):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + norm*0.01,
                    f'{norm:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gradient flow saved to: {save_path}")

        plt.show()
        return plt.gcf()


class ActivationAnalyzer:
    """Analyze and visualize activation function behaviors."""

    def __init__(self, save_dir='./visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_activation_functions(self, activation_names, x_range=(-5, 5), save_path=None):
        """
        Plot multiple activation functions for comparison.

        Args:
            activation_names (list): List of activation function names
            x_range (tuple): Range of x values to plot
            save_path (str, optional): Path to save the plot
        """
        from models import get_activation

        x = torch.linspace(x_range[0], x_range[1], 1000)

        plt.figure(figsize=(15, 10))

        # Create subplots
        n_activations = len(activation_names)
        cols = 3
        rows = (n_activations + cols - 1) // cols

        for i, act_name in enumerate(activation_names):
            plt.subplot(rows, cols, i + 1)

            try:
                activation_fn = get_activation(act_name)

                # Handle special cases
                if act_name.lower() == 'softmax':
                    # For softmax, use a different approach
                    x_2d = x.unsqueeze(0)  # Add batch dimension
                    y = activation_fn(x_2d).squeeze(0)
                else:
                    y = activation_fn(x)

                y_np = y.detach().numpy()
                x_np = x.detach().numpy()

                plt.plot(x_np, y_np, linewidth=2, label=act_name.upper())
                plt.title(f'{act_name.upper()} Activation', fontweight='bold')
                plt.xlabel('Input (x)')
                plt.ylabel('Output f(x)')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

                # Add derivative if possible
                if act_name.lower() not in ['step', 'softmax']:
                    x.requires_grad_(True)
                    y_grad = activation_fn(x)
                    y_grad.sum().backward()
                    derivative = x.grad.detach().numpy()

                    plt.plot(x_np, derivative, '--', alpha=0.7,
                            label=f"{act_name} derivative")
                    plt.legend()
                    x.grad.zero_()
                    x.requires_grad_(False)

            except Exception as e:
                plt.text(0.5, 0.5, f'Error: {str(e)}',
                        transform=plt.gca().transAxes, ha='center', va='center')
                plt.title(f'{act_name.upper()} - Error')

        plt.suptitle('Activation Functions Comparison', fontsize=16, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Activation functions plot saved to: {save_path}")

        plt.show()
        return plt.gcf()

    def analyze_activation_properties(self, activation_names):
        """
        Analyze mathematical properties of activation functions.

        Args:
            activation_names (list): List of activation function names

        Returns:
            dict: Properties of each activation function
        """
        from models import get_activation

        properties = {}
        x_test = torch.linspace(-10, 10, 1000)

        for act_name in activation_names:
            try:
                activation_fn = get_activation(act_name)

                if act_name.lower() == 'softmax':
                    continue  # Skip softmax for property analysis

                y = activation_fn(x_test)
                y_np = y.detach().numpy()

                properties[act_name] = {
                    'range': (float(y_np.min()), float(y_np.max())),
                    'mean_output': float(y_np.mean()),
                    'zero_centered': abs(y_np.mean()) < 0.1,
                    'monotonic': all(y_np[i] <= y_np[i+1] for i in range(len(y_np)-1)),
                    'bounded': np.isfinite(y_np.max()) and np.isfinite(y_np.min()),
                    'saturating': y_np.max() - y_np.min() < 20  # Heuristic for saturation
                }

            except Exception as e:
                properties[act_name] = {'error': str(e)}

        return properties


class RealTimeMonitor:
    """Monitor training progress in real-time."""

    def __init__(self):
        self.training_data = defaultdict(list)
        self.epoch_data = defaultdict(list)

    def update(self, epoch, batch, loss, accuracy, lr=None):
        """Update monitoring data."""
        self.training_data['epoch'].append(epoch)
        self.training_data['batch'].append(batch)
        self.training_data['loss'].append(loss)
        self.training_data['accuracy'].append(accuracy)
        if lr is not None:
            self.training_data['lr'].append(lr)

    def plot_training_progress(self, save_path=None):
        """Plot real-time training progress."""
        if not self.training_data['loss']:
            print("No training data to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curve
        axes[0, 0].plot(self.training_data['loss'], color='red', alpha=0.7)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curve
        axes[0, 1].plot(self.training_data['accuracy'], color='blue', alpha=0.7)
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate (if available)
        if 'lr' in self.training_data:
            axes[1, 0].plot(self.training_data['lr'], color='green', alpha=0.7)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # Loss distribution
        axes[1, 1].hist(self.training_data['loss'], bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Real-time Training Monitor', fontsize=16, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig