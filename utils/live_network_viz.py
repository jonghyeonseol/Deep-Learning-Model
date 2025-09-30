import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import time
from collections import defaultdict, deque
import threading
import queue

# Enable interactive plotting
plt.ion()


class LiveNetworkVisualizer:
    """
    Real-time visualization of neural network with circles (neurons) and arrows (connections).
    Shows live neuron activations, weight changes, and data flow during training.
    """

    def __init__(self, model, max_neurons_per_layer=8, update_frequency=20):
        """
        Initialize live network visualizer.

        Args:
            model: PyTorch model to visualize
            max_neurons_per_layer: Maximum neurons to show per layer
            update_frequency: Update every N batches
        """
        self.model = model
        self.max_neurons_per_layer = max_neurons_per_layer
        self.update_frequency = update_frequency

        # Network structure
        self.layers = []
        self.connections = []
        self.neurons = {}
        self.weights = {}

        # Animation data
        self.activations = {}
        self.current_activations = {}
        self.weight_changes = {}
        self.data_flow = deque(maxlen=10)

        # Control variables
        self.is_running = False
        self.batch_count = 0
        self.hooks = []

        # Setup
        self.fig = None
        self.ax = None
        self.animation = None

        self._analyze_network_structure()
        self._setup_hooks()

    def _analyze_network_structure(self):
        """Analyze the network structure to create visualization layout."""
        layer_info = []

        # Get layer information
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_info.append({
                    'name': name,
                    'type': 'Linear',
                    'input_size': module.in_features,
                    'output_size': module.out_features,
                    'module': module
                })
            elif isinstance(module, nn.Conv2d):
                layer_info.append({
                    'name': name,
                    'type': 'Conv2d',
                    'input_channels': module.in_channels,
                    'output_channels': module.out_channels,
                    'module': module
                })

        self.layers = layer_info
        print(f"🧠 Network structure: {len(self.layers)} layers detected")

    def _setup_hooks(self):
        """Setup hooks to capture activations during forward pass."""
        def create_hook(layer_name):
            def hook(module, input, output):
                if self.is_running and isinstance(output, torch.Tensor):
                    # Store activation data
                    with torch.no_grad():
                        if len(output.shape) == 4:  # Conv layer (batch, channels, height, width)
                            # Average pool the spatial dimensions
                            data = output.mean(dim=[2, 3]).cpu().numpy()
                        else:  # Linear layer (batch, features)
                            data = output.cpu().numpy()

                        # Store mean activation for each neuron
                        mean_activations = np.mean(data, axis=0)
                        self.current_activations[layer_name] = mean_activations
            return hook

        # Register hooks
        for layer_info in self.layers:
            hook = create_hook(layer_info['name'])
            self.hooks.append(layer_info['module'].register_forward_hook(hook))

    def start_visualization(self):
        """Start the live network visualization."""
        self.is_running = True
        self.batch_count = 0
        self._setup_visualization()
        print("🎨 Live network visualization STARTED")
        print("📺 Watch neurons (circles) light up and connections (arrows) change!")

    def stop_visualization(self):
        """Stop the live network visualization."""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        print("⏹️ Live network visualization STOPPED")

    def _setup_visualization(self):
        """Setup the matplotlib figure and network layout."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 10))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('🧠 Live Neural Network - Circles & Arrows Visualization',
                         fontsize=16, fontweight='bold', pad=20)

        # Calculate layer positions
        n_layers = len(self.layers)
        if n_layers == 0:
            return

        layer_x_positions = np.linspace(1, 9, n_layers + 1)  # +1 for input

        # Create network layout
        self._create_network_layout(layer_x_positions)

        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self._animate_network, interval=100, blit=False, cache_frame_data=False
        )

        plt.show(block=False)

    def _create_network_layout(self, layer_x_positions):
        """Create the initial network layout with circles and arrows."""
        # Input layer
        input_x = layer_x_positions[0]
        if self.layers and self.layers[0]['type'] == 'Linear':
            input_size = min(self.layers[0]['input_size'], self.max_neurons_per_layer)
        else:
            input_size = min(3, self.max_neurons_per_layer)  # For conv networks

        input_y_positions = np.linspace(2, 6, input_size)

        # Store neuron positions
        self.neurons['input'] = []
        for i, y_pos in enumerate(input_y_positions):
            neuron = {
                'circle': Circle((input_x, y_pos), 0.15, color='lightblue', ec='black', linewidth=2),
                'position': (input_x, y_pos),
                'activation': 0.0,
                'id': f'input_{i}'
            }
            self.neurons['input'].append(neuron)
            self.ax.add_patch(neuron['circle'])

        # Add input label
        self.ax.text(input_x, 1.5, 'INPUT\nLAYER', ha='center', va='top',
                    fontsize=10, fontweight='bold')

        # Hidden and output layers
        for layer_idx, layer_info in enumerate(self.layers):
            layer_x = layer_x_positions[layer_idx + 1]

            if layer_info['type'] == 'Linear':
                n_neurons = min(layer_info['output_size'], self.max_neurons_per_layer)
            else:  # Conv2d
                n_neurons = min(layer_info['output_channels'], self.max_neurons_per_layer)

            neuron_y_positions = np.linspace(2, 6, n_neurons)

            # Create neurons for this layer
            layer_name = layer_info['name']
            self.neurons[layer_name] = []

            for i, y_pos in enumerate(neuron_y_positions):
                # Color based on layer type
                if layer_idx == len(self.layers) - 1:  # Output layer
                    color = 'lightgreen'
                    label = 'OUTPUT'
                else:  # Hidden layer
                    color = 'lightcoral'
                    label = f'LAYER {layer_idx + 1}'

                neuron = {
                    'circle': Circle((layer_x, y_pos), 0.15, color=color, ec='black', linewidth=2),
                    'position': (layer_x, y_pos),
                    'activation': 0.0,
                    'id': f'{layer_name}_{i}'
                }
                self.neurons[layer_name].append(neuron)
                self.ax.add_patch(neuron['circle'])

            # Add layer label
            if layer_idx == len(self.layers) - 1:
                self.ax.text(layer_x, 1.5, f'{label}\nLAYER', ha='center', va='top',
                            fontsize=10, fontweight='bold')
            else:
                self.ax.text(layer_x, 1.5, f'HIDDEN\n{label}', ha='center', va='top',
                            fontsize=10, fontweight='bold')

        # Create connections (arrows)
        self._create_connections(layer_x_positions)

        # Add legend
        self._add_legend()

    def _create_connections(self, layer_x_positions):
        """Create arrows showing connections between layers."""
        self.connections = []

        prev_layer_neurons = self.neurons['input']

        for layer_idx, layer_info in enumerate(self.layers):
            layer_name = layer_info['name']
            current_layer_neurons = self.neurons[layer_name]

            # Create connections between previous and current layer
            for prev_neuron in prev_layer_neurons:
                for curr_neuron in current_layer_neurons:
                    # Sample some connections (not all, would be too crowded)
                    if np.random.random() < 0.3:  # Show 30% of connections
                        arrow = FancyArrowPatch(
                            prev_neuron['position'], curr_neuron['position'],
                            arrowstyle='->', mutation_scale=15, alpha=0.3,
                            color='gray', linewidth=1
                        )
                        self.ax.add_patch(arrow)

                        connection = {
                            'arrow': arrow,
                            'from': prev_neuron['id'],
                            'to': curr_neuron['id'],
                            'weight': np.random.normal(0, 0.1)  # Initial random weight
                        }
                        self.connections.append(connection)

            prev_layer_neurons = current_layer_neurons

    def _add_legend(self):
        """Add legend explaining the visualization."""
        legend_text = """
🔵 Input Neurons    🔴 Hidden Neurons    🟢 Output Neurons
Brightness = Activation Level    Arrow Thickness = Weight Strength
"""
        self.ax.text(5, 0.5, legend_text, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    def update(self):
        """Update the visualization with current network state."""
        if not self.is_running:
            return

        self.batch_count += 1

    def _animate_network(self, frame):
        """Animation function called by matplotlib."""
        if not self.is_running or not self.current_activations:
            return

        # Update neuron colors based on activations
        for layer_name, activations in self.current_activations.items():
            if layer_name in self.neurons:
                layer_neurons = self.neurons[layer_name]

                # Normalize activations for coloring
                if len(activations) > 0:
                    max_activation = max(abs(np.max(activations)), abs(np.min(activations)), 0.001)

                    for i, neuron in enumerate(layer_neurons):
                        if i < len(activations):
                            # Normalize activation to [0, 1]
                            normalized_activation = abs(activations[i]) / max_activation
                            normalized_activation = min(normalized_activation, 1.0)

                            # Update neuron color intensity
                            if layer_name == list(self.neurons.keys())[-1]:  # Output layer
                                base_color = 'green'
                            elif layer_name == 'input':
                                base_color = 'blue'
                            else:
                                base_color = 'red'

                            # Create color with varying intensity
                            intensity = 0.3 + 0.7 * normalized_activation  # 0.3 to 1.0
                            if base_color == 'blue':
                                color = (0.7, 0.7 + 0.3 * intensity, 1.0)
                            elif base_color == 'red':
                                color = (1.0, 0.7 + 0.3 * intensity, 0.7)
                            else:  # green
                                color = (0.7, 1.0, 0.7 + 0.3 * intensity)

                            neuron['circle'].set_facecolor(color)
                            neuron['activation'] = normalized_activation

        # Update connection weights (simplified)
        for connection in self.connections:
            # Simulate weight changes
            connection['weight'] += np.random.normal(0, 0.01)

            # Update arrow appearance based on weight
            weight_magnitude = abs(connection['weight'])
            alpha = min(0.8, 0.2 + weight_magnitude * 2)
            linewidth = max(0.5, min(3, weight_magnitude * 10))

            connection['arrow'].set_alpha(alpha)
            connection['arrow'].set_linewidth(linewidth)

            # Color based on weight sign
            color = 'blue' if connection['weight'] > 0 else 'red'
            connection['arrow'].set_color(color)

        # Add data flow animation (optional pulsing effect)
        current_time = time.time()
        pulse = 0.5 + 0.5 * np.sin(current_time * 3)  # Pulsing effect

        # Apply pulse to active connections
        for connection in self.connections[:10]:  # Animate first 10 connections
            current_alpha = connection['arrow'].get_alpha()
            pulsed_alpha = current_alpha * pulse
            connection['arrow'].set_alpha(pulsed_alpha)

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class LivePerceptronNetwork:
    """
    Simplified live visualization of perceptron network structure.
    Shows a clear network diagram with circles and arrows.
    """

    def __init__(self, network_structure=[3, 4, 4, 2]):
        """
        Initialize live perceptron network visualization.

        Args:
            network_structure: List of neurons per layer [input, hidden1, hidden2, output]
        """
        self.structure = network_structure
        self.fig = None
        self.ax = None
        self.neurons = {}
        self.connections = []
        self.is_running = False

    def start_visualization(self):
        """Start the live perceptron network visualization."""
        self.is_running = True
        self._setup_network()
        print("🎨 Live perceptron network STARTED")

    def _setup_network(self):
        """Setup the perceptron network visualization."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 8))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 6)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('🧠 Live Perceptron Network - Circles & Arrows',
                         fontsize=16, fontweight='bold', pad=20)

        # Calculate layer positions
        n_layers = len(self.structure)
        layer_x_positions = np.linspace(1, 9, n_layers)

        # Create layers
        for layer_idx, n_neurons in enumerate(self.structure):
            layer_x = layer_x_positions[layer_idx]

            # Calculate neuron positions for this layer
            if n_neurons == 1:
                neuron_y_positions = [3]  # Center for single neuron
            else:
                neuron_y_positions = np.linspace(1.5, 4.5, n_neurons)

            # Create neurons
            layer_name = f'layer_{layer_idx}'
            self.neurons[layer_name] = []

            for neuron_idx, y_pos in enumerate(neuron_y_positions):
                # Color coding
                if layer_idx == 0:  # Input layer
                    color = 'lightblue'
                    label_prefix = 'X'
                elif layer_idx == len(self.structure) - 1:  # Output layer
                    color = 'lightgreen'
                    label_prefix = 'Y'
                else:  # Hidden layers
                    color = 'lightcoral'
                    label_prefix = 'H'

                # Create neuron circle
                circle = Circle((layer_x, y_pos), 0.2, color=color, ec='black', linewidth=2)
                self.ax.add_patch(circle)

                # Add neuron label
                if layer_idx == 0 or layer_idx == len(self.structure) - 1:
                    label = f'{label_prefix}{neuron_idx + 1}'
                else:
                    label = f'{label_prefix}{layer_idx}.{neuron_idx + 1}'

                self.ax.text(layer_x, y_pos, label, ha='center', va='center',
                           fontsize=8, fontweight='bold')

                neuron = {
                    'circle': circle,
                    'position': (layer_x, y_pos),
                    'activation': 0.0,
                    'layer': layer_idx,
                    'index': neuron_idx
                }
                self.neurons[layer_name].append(neuron)

            # Add layer labels
            if layer_idx == 0:
                self.ax.text(layer_x, 0.8, 'INPUT\nLAYER', ha='center', va='center',
                           fontsize=10, fontweight='bold')
            elif layer_idx == len(self.structure) - 1:
                self.ax.text(layer_x, 0.8, 'OUTPUT\nLAYER', ha='center', va='center',
                           fontsize=10, fontweight='bold')
            else:
                self.ax.text(layer_x, 0.8, f'HIDDEN\nLAYER {layer_idx}', ha='center', va='center',
                           fontsize=10, fontweight='bold')

        # Create connections
        self._create_all_connections()

        # Add legend and information
        self._add_network_info()

        plt.show(block=False)

    def _create_all_connections(self):
        """Create all connections between adjacent layers."""
        layer_names = list(self.neurons.keys())

        for i in range(len(layer_names) - 1):
            from_layer = self.neurons[layer_names[i]]
            to_layer = self.neurons[layer_names[i + 1]]

            for from_neuron in from_layer:
                for to_neuron in to_layer:
                    # Create arrow
                    arrow = FancyArrowPatch(
                        from_neuron['position'], to_neuron['position'],
                        arrowstyle='->', mutation_scale=12, alpha=0.6,
                        color='gray', linewidth=1.5
                    )
                    self.ax.add_patch(arrow)

                    connection = {
                        'arrow': arrow,
                        'from_neuron': from_neuron,
                        'to_neuron': to_neuron,
                        'weight': np.random.normal(0, 0.5)
                    }
                    self.connections.append(connection)

    def _add_network_info(self):
        """Add network information and legend."""
        info_text = f"""
🧠 Perceptron Network Structure: {' → '.join(map(str, self.structure))}
🔵 Input Neurons  🔴 Hidden Neurons  🟢 Output Neurons
➡️ Connections (Weights)  💡 Activation Flow
"""
        self.ax.text(5, 5.5, info_text, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))

    def animate_data_flow(self, input_data=None):
        """Animate data flowing through the network."""
        if not self.is_running:
            return

        # Simulate forward pass
        if input_data is None:
            input_data = np.random.randn(self.structure[0])

        # Update input layer
        input_layer = self.neurons['layer_0']
        for i, neuron in enumerate(input_layer):
            if i < len(input_data):
                activation = abs(input_data[i])
                neuron['activation'] = activation
                # Update color intensity
                intensity = min(1.0, activation)
                color = (0.5 + 0.5 * intensity, 0.5 + 0.5 * intensity, 1.0)
                neuron['circle'].set_facecolor(color)

        # Simulate activation propagation
        layer_names = list(self.neurons.keys())
        for i in range(1, len(layer_names)):
            layer = self.neurons[layer_names[i]]
            for neuron in layer:
                # Simulate activation
                activation = np.random.random()
                neuron['activation'] = activation

                # Update color based on activation
                if i == len(layer_names) - 1:  # Output layer
                    color = (0.5 + 0.5 * activation, 1.0, 0.5 + 0.5 * activation)
                else:  # Hidden layer
                    color = (1.0, 0.5 + 0.5 * activation, 0.5 + 0.5 * activation)

                neuron['circle'].set_facecolor(color)

        # Update connection weights visualization
        for connection in self.connections:
            weight = connection['weight']
            weight_magnitude = abs(weight)

            # Update arrow appearance
            alpha = min(0.8, 0.3 + weight_magnitude)
            linewidth = max(0.5, min(3, weight_magnitude * 2))

            connection['arrow'].set_alpha(alpha)
            connection['arrow'].set_linewidth(linewidth)

            # Color based on weight sign
            color = 'blue' if weight > 0 else 'red'
            connection['arrow'].set_color(color)

        plt.draw()
        plt.pause(0.1)

    def stop_visualization(self):
        """Stop the visualization."""
        self.is_running = False
        if self.fig:
            plt.close(self.fig)


def create_live_network_demo():
    """Create a demo of live network visualization."""
    print("🎨 Creating live perceptron network visualization...")

    # Create a simple network structure
    network = LivePerceptronNetwork([3, 5, 4, 2])
    network.start_visualization()

    print("🎬 Starting animation... (Press Ctrl+C to stop)")

    try:
        for i in range(100):  # Run for 100 iterations
            print(f"📊 Iteration {i+1}/100 - Watch the network process data!")

            # Simulate different input patterns
            if i % 20 == 0:
                input_data = [1, 0, 0]  # Pattern 1
            elif i % 20 == 10:
                input_data = [0, 1, 0]  # Pattern 2
            else:
                input_data = np.random.randn(3)  # Random data

            network.animate_data_flow(input_data)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n⏹️ Animation stopped by user")

    network.stop_visualization()
    print("✅ Demo completed!")


if __name__ == '__main__':
    create_live_network_demo()