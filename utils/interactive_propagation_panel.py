"""
Interactive Propagation Panel - Real-time visualization of forward and backward propagation.
Shows step-by-step neuron interactions through synapses with detailed weight visualization.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.widgets import Button, Slider
import numpy as np
import time
from collections import deque


class InteractivePropagationPanel:
    """
    Interactive panel showing neuron-synapse interactions during propagation.
    Features:
    - Detailed synapse visualization with weight values
    - Step-by-step signal flow through connections
    - Weighted sum calculation at each neuron
    - Activation function visualization
    - Gradient flow through synapses (backprop)
    """

    def __init__(self, model, sample_input, sample_target):
        """
        Initialize interactive propagation panel.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor for visualization
            sample_target: Sample target for loss calculation
        """
        self.model = model
        self.sample_input = sample_input
        self.sample_target = sample_target
        self.device = next(model.parameters()).device

        # Animation state
        self.is_running = False
        self.is_paused = False
        self.animation_speed = 0.8  # seconds per step
        self.current_step = 0
        self.mode = 'forward'  # 'forward' or 'backward'

        # Network data
        self.layers = []
        self.layer_activations = {}
        self.layer_gradients = {}
        self.layer_weights = {}
        self.hooks = []

        # Visualization
        self.fig = None
        self.ax_main = None
        self.ax_info = None
        self.neurons = {}
        self.synapses = []  # Synapse objects with weights
        self.info_text = None
        self.synapse_labels = []  # Text labels for synapse weights

        # Controls
        self.btn_pause = None
        self.btn_step = None
        self.btn_reset = None
        self.speed_slider = None

        self._analyze_network()
        self._setup_hooks()

    def _analyze_network(self):
        """Analyze network structure and extract weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.layers.append({
                    'name': name,
                    'module': module,
                    'type': type(module).__name__
                })
                # Store weight information
                with torch.no_grad():
                    weights = module.weight.cpu().numpy()
                    self.layer_weights[name] = weights

        print(f"📊 Panel initialized: {len(self.layers)} layers detected")
        for i, layer in enumerate(self.layers):
            print(f"   Layer {i+1}: {layer['name']} ({layer['type']})")

    def _setup_hooks(self):
        """Setup hooks to capture activations and gradients."""

        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    with torch.no_grad():
                        # Store activation data
                        if len(output.shape) == 4:  # Conv layer
                            data = output.mean(dim=[2, 3])
                        else:
                            data = output
                        self.layer_activations[name] = {
                            'data': data.cpu().numpy(),
                            'mean': float(data.mean()),
                            'std': float(data.std()),
                            'max': float(data.max()),
                            'min': float(data.min())
                        }
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    with torch.no_grad():
                        grad = grad_output[0]
                        if len(grad.shape) == 4:  # Conv layer
                            grad_data = grad.mean(dim=[2, 3])
                        else:
                            grad_data = grad
                        self.layer_gradients[name] = {
                            'data': grad_data.cpu().numpy(),
                            'mean': float(grad_data.mean()),
                            'std': float(grad_data.std()),
                            'max': float(grad_data.max()),
                            'min': float(grad_data.min()),
                            'norm': float(grad.norm())
                        }
            return hook

        for layer in self.layers:
            name = layer['name']
            module = layer['module']
            self.hooks.append(module.register_forward_hook(forward_hook(name)))
            self.hooks.append(module.register_full_backward_hook(backward_hook(name)))

    def start_panel(self):
        """Start the interactive propagation panel."""
        self.is_running = True
        self._setup_panel()
        self._run_animation()
        plt.show()

    def _setup_panel(self):
        """Setup the panel interface with emphasis on synapses."""
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.suptitle('🧠 Interactive Neural Network Propagation Panel\n'
                         'Synapse Interactions & Signal Flow Visualization',
                         fontsize=16, fontweight='bold')

        # Main visualization area (larger)
        self.ax_main = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=4)
        self.ax_main.set_xlim(0, 12)
        self.ax_main.set_ylim(0, 10)
        self.ax_main.axis('off')

        # Info panel
        self.ax_info = plt.subplot2grid((5, 3), (4, 0), colspan=2)
        self.ax_info.axis('off')

        # Control panel
        ax_controls = plt.subplot2grid((5, 3), (4, 2))
        ax_controls.axis('off')

        # Add control buttons
        ax_btn_pause = plt.axes([0.68, 0.08, 0.08, 0.04])
        ax_btn_step = plt.axes([0.77, 0.08, 0.08, 0.04])
        ax_btn_reset = plt.axes([0.86, 0.08, 0.08, 0.04])
        ax_speed = plt.axes([0.68, 0.02, 0.26, 0.03])

        self.btn_pause = Button(ax_btn_pause, '⏸ Pause', color='lightcoral')
        self.btn_pause.on_clicked(self._on_pause)

        self.btn_step = Button(ax_btn_step, '⏭ Step', color='lightblue')
        self.btn_step.on_clicked(self._on_step)

        self.btn_reset = Button(ax_btn_reset, '🔄 Reset', color='lightgreen')
        self.btn_reset.on_clicked(self._on_reset)

        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 2.0, valinit=0.8)
        self.speed_slider.on_changed(self._on_speed_change)

        # Draw network with synapses
        self._draw_network_with_synapses()

        # Initialize info text
        self.info_text = self.ax_info.text(0.05, 0.5, '', transform=self.ax_info.transAxes,
                                          fontsize=9, verticalalignment='center',
                                          family='monospace')

        plt.tight_layout()

    def _draw_network_with_synapses(self):
        """Draw the neural network with detailed synapse visualization."""
        # Limit neurons for clarity (show representative connections)
        max_neurons = 4
        # Show all layers (input + all model layers)
        n_layers = len(self.layers) + 1  # +1 for input layer
        layer_x = np.linspace(1.5, 10.5, n_layers)

        # Input layer
        input_neurons = min(max_neurons, 3)
        input_y = np.linspace(4, 6, input_neurons)
        self.neurons['input'] = []

        for i, y in enumerate(input_y):
            circle = Circle((layer_x[0], y), 0.25, color='lightblue',
                          ec='black', linewidth=2.5, zorder=5)
            self.ax_main.add_patch(circle)

            # Add neuron label
            self.ax_main.text(layer_x[0], y, f'x{i+1}', ha='center', va='center',
                            fontsize=9, fontweight='bold', zorder=6)

            self.neurons['input'].append({
                'circle': circle,
                'pos': (layer_x[0], y),
                'activation': 0,
                'id': f'input_{i}'
            })

        # Layer label with box
        label_box = FancyBboxPatch((layer_x[0]-0.5, 2.8), 1, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightblue', edgecolor='black',
                                   linewidth=2, alpha=0.3)
        self.ax_main.add_patch(label_box)
        self.ax_main.text(layer_x[0], 3.1, 'INPUT\nLAYER', ha='center', va='center',
                         fontsize=10, fontweight='bold')

        # Hidden and output layers
        prev_neurons = self.neurons['input']

        # Show ALL layers
        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            x = layer_x[idx + 1]
            n_neurons = min(max_neurons, self._get_layer_size(layer))
            neuron_y = np.linspace(4, 6, n_neurons)

            # Determine color and label - check if this is the LAST layer
            if idx == len(self.layers) - 1:
                color = 'lightgreen'
                label = 'OUTPUT\nLAYER'
                box_color = 'lightgreen'
            else:
                color = 'lightcoral'
                label = f'HIDDEN\nLAYER {idx + 1}'
                box_color = 'lightcoral'

            layer_name = layer['name']
            self.neurons[layer_name] = []

            # Create neurons
            for i, y in enumerate(neuron_y):
                circle = Circle((x, y), 0.25, color=color,
                              ec='black', linewidth=2.5, zorder=5)
                self.ax_main.add_patch(circle)

                # Add activation function symbol
                self.ax_main.text(x, y, 'σ', ha='center', va='center',
                                fontsize=11, fontweight='bold', zorder=6,
                                style='italic')

                self.neurons[layer_name].append({
                    'circle': circle,
                    'pos': (x, y),
                    'activation': 0,
                    'gradient': 0,
                    'id': f'{layer_name}_{i}'
                })

            # Layer label with box
            label_box = FancyBboxPatch((x-0.5, 2.8), 1, 0.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=box_color, edgecolor='black',
                                       linewidth=2, alpha=0.3)
            self.ax_main.add_patch(label_box)
            self.ax_main.text(x, 3.1, label, ha='center', va='center',
                             fontsize=10, fontweight='bold')

            # Draw SYNAPSES with weight information
            self._draw_synapses(prev_neurons, self.neurons[layer_name], layer_name)

            prev_neurons = self.neurons[layer_name]

        # Add mode indicators
        self.forward_indicator = FancyBboxPatch((0.3, 8), 2, 0.8,
                                               boxstyle="round,pad=0.1",
                                               facecolor='green', alpha=0.5,
                                               edgecolor='black', linewidth=3)
        self.backward_indicator = FancyBboxPatch((9.7, 8), 2, 0.8,
                                                boxstyle="round,pad=0.1",
                                                facecolor='red', alpha=0.0,
                                                edgecolor='black', linewidth=3)
        self.ax_main.add_patch(self.forward_indicator)
        self.ax_main.add_patch(self.backward_indicator)
        self.ax_main.text(1.3, 8.4, '→ FORWARD\nPROPAGATION', ha='center', va='center',
                         fontsize=11, fontweight='bold')
        self.ax_main.text(10.7, 8.4, '← BACKWARD\nPROPAGATION', ha='center', va='center',
                         fontsize=11, fontweight='bold')

        # Add legend explaining synapses
        legend_text = (
            "🔵 Synapse (Connection)  |  📊 Weight Value  |  "
            "💡 Neuron Activation  |  🎯 Gradient Flow"
        )
        legend_box = FancyBboxPatch((0.5, 0.3), 11, 0.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightyellow', edgecolor='black',
                                   linewidth=2, alpha=0.8)
        self.ax_main.add_patch(legend_box)
        self.ax_main.text(6, 0.6, legend_text, ha='center', va='center',
                         fontsize=10, fontweight='bold')

    def _draw_synapses(self, from_neurons, to_neurons, layer_name):
        """Draw synapses (connections) with weight visualization."""
        # Get weights for this layer
        if layer_name in self.layer_weights:
            weights = self.layer_weights[layer_name]
        else:
            weights = None

        # Draw connections between neurons
        for i, from_neuron in enumerate(from_neurons):
            for j, to_neuron in enumerate(to_neurons):
                # Get weight value if available
                if weights is not None and j < weights.shape[0]:
                    # Handle different weight tensor shapes
                    if len(weights.shape) == 4:  # Conv2d: (out_ch, in_ch, h, w)
                        if i < weights.shape[1]:
                            # Average over spatial dimensions for conv kernels
                            weight = float(weights[j, i].mean())
                        else:
                            weight = np.random.normal(0, 0.5)
                    elif len(weights.shape) == 2:  # Linear: (out_features, in_features)
                        if i < weights.shape[1]:
                            weight = float(weights[j, i])
                        else:
                            weight = np.random.normal(0, 0.5)
                    else:
                        weight = np.random.normal(0, 0.5)
                else:
                    weight = np.random.normal(0, 0.5)  # Random for visualization

                # Create synapse arrow
                arrow = FancyArrowPatch(
                    from_neuron['pos'], to_neuron['pos'],
                    arrowstyle='->', mutation_scale=20,
                    color='gray', alpha=0.4, linewidth=2, zorder=2
                )
                self.ax_main.add_patch(arrow)

                # Calculate midpoint for weight label
                mid_x = (from_neuron['pos'][0] + to_neuron['pos'][0]) / 2
                mid_y = (from_neuron['pos'][1] + to_neuron['pos'][1]) / 2

                # Add weight label (small text box)
                weight_text = self.ax_main.text(
                    mid_x, mid_y, f'{weight:.2f}',
                    ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.7),
                    zorder=3, visible=False  # Initially hidden
                )

                synapse = {
                    'arrow': arrow,
                    'from': from_neuron,
                    'to': to_neuron,
                    'weight': weight,
                    'weight_label': weight_text,
                    'layer': layer_name
                }
                self.synapses.append(synapse)

    def _get_layer_size(self, layer):
        """Get the output size of a layer."""
        module = layer['module']
        if isinstance(module, nn.Linear):
            return module.out_features
        elif isinstance(module, nn.Conv2d):
            return module.out_channels
        return 4

    def _run_animation(self):
        """Run the propagation animation."""
        print("\n" + "="*70)
        print("🎬 INTERACTIVE PROPAGATION PANEL STARTED")
        print("="*70)
        print("\n📺 Controls:")
        print("   ⏸  Pause/Resume: Pause or resume the animation")
        print("   ⏭  Step: Execute one step at a time (auto-pauses)")
        print("   🔄 Reset: Restart from forward propagation")
        print("   🎚  Speed Slider: Adjust animation speed (0.1-2.0s per step)\n")
        print("🔍 What You're Seeing:")
        print("   🔵 Circles = Neurons (process incoming signals)")
        print("   ➡️  Arrows = Synapses (weighted connections)")
        print("   📊 Numbers on arrows = Synapse weights (w)")
        print("   💡 Neuron color brightness = Activation level")
        print("   🔴 Red synapses = Negative weights")
        print("   🔵 Blue synapses = Positive weights")
        print("   🟢 Green highlight = Forward pass active")
        print("   🔴 Red highlight = Backward pass active\n")
        print("📐 Mathematics:")
        print("   Forward:  neuron_output = σ(Σ(input_i × weight_i) + bias)")
        print("   Backward: ∂Loss/∂weight = ∂Loss/∂output × ∂output/∂weight\n")

        iteration = 0
        while self.is_running:
            if not self.is_paused:
                iteration += 1
                print(f"\n{'='*70}")
                print(f"🔄 Iteration {iteration}")
                print(f"{'='*70}")

                # Run forward pass
                self._execute_forward_pass()

                time.sleep(1)  # Brief pause between forward and backward

                # Run backward pass
                self._execute_backward_pass()

                time.sleep(1.5)  # Pause between iterations

            plt.pause(0.01)

            # Check if figure is closed
            if not plt.fignum_exists(self.fig.number):
                self.is_running = False
                break

    def _execute_forward_pass(self):
        """Execute and visualize forward pass with synapse interactions."""
        self.mode = 'forward'
        self.forward_indicator.set_alpha(0.6)
        self.backward_indicator.set_alpha(0.0)

        print("\n→ FORWARD PROPAGATION - Signal flows through synapses:")

        # Clear previous data
        self.layer_activations.clear()

        # Move sample to device
        input_data = self.sample_input.to(self.device)
        target = self.sample_target.to(self.device)

        # Simulate input activation
        input_values = input_data[0].cpu().numpy().flatten()[:len(self.neurons['input'])]
        for i, neuron in enumerate(self.neurons['input']):
            if i < len(input_values):
                val = float(input_values[i])
                neuron['activation'] = abs(val)
                intensity = min(1.0, abs(val))
                color = (0.5 + 0.5*intensity, 0.5 + 0.5*intensity, 1.0)
                neuron['circle'].set_facecolor(color)

        plt.draw()
        plt.pause(0.01)
        time.sleep(self.animation_speed * 0.5)

        # Forward pass through model
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)

        # Animate layer by layer with synapse visualization
        layer_keys = [k for k in self.neurons.keys() if k != 'input']

        for idx, layer_name in enumerate(layer_keys):
            if layer_name in self.layer_activations:
                self._animate_layer_forward_with_synapses(layer_name, idx)
                time.sleep(self.animation_speed)

        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        print(f"\n   ✅ Forward pass complete!")
        print(f"   📊 Final output computed")
        print(f"   📉 Loss: {loss.item():.6f}")

    def _execute_backward_pass(self):
        """Execute and visualize backward pass with gradient flow through synapses."""
        self.mode = 'backward'
        self.forward_indicator.set_alpha(0.0)
        self.backward_indicator.set_alpha(0.6)

        print("\n← BACKWARD PROPAGATION - Gradients flow back through synapses:")

        # Clear previous gradients
        self.layer_gradients.clear()
        self.model.zero_grad()

        # Forward pass
        input_data = self.sample_input.to(self.device)
        target = self.sample_target.to(self.device)

        self.model.train()
        output = self.model(input_data)

        # Backward pass
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()

        # Animate layer by layer (reverse order)
        layer_keys = [k for k in self.neurons.keys() if k != 'input']

        for idx, layer_name in enumerate(reversed(layer_keys)):
            if layer_name in self.layer_gradients:
                self._animate_layer_backward_with_synapses(layer_name,
                                                          len(layer_keys) - idx - 1)
                time.sleep(self.animation_speed)

        print(f"\n   ✅ Backward pass complete!")
        print(f"   🎯 Gradients computed for all weights")

        # Calculate total gradient norm
        total_grad_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"   📊 Total gradient norm: {total_grad_norm:.6f}")

    def _animate_layer_forward_with_synapses(self, layer_name, layer_idx):
        """Animate forward propagation showing synapse interactions."""
        if layer_name not in self.layer_activations:
            return

        activation_data = self.layer_activations[layer_name]
        neurons = self.neurons[layer_name]

        print(f"\n   🧠 Layer {layer_idx + 1} ({layer_name})")
        print(f"      Computing: output = σ(Σ(input × weight) + bias)")

        # Update info panel with mathematical detail
        info = f"""
FORWARD PASS - Layer {layer_idx + 1}: {layer_name}
{'='*60}
SYNAPSE INTERACTION:
  Each neuron receives weighted signals through synapses:

  neuron_input = Σ(previous_activation_i × synapse_weight_i)
  neuron_output = activation_function(neuron_input + bias)

ACTIVATION STATISTICS:
  Mean:  {activation_data['mean']:>8.4f}    (average activation level)
  Std:   {activation_data['std']:>8.4f}    (variation in activations)
  Max:   {activation_data['max']:>8.4f}    (strongest activation)
  Min:   {activation_data['min']:>8.4f}    (weakest activation)

➡️  Watch blue synapses carry signals forward!
"""
        self.info_text.set_text(info)

        # Highlight and show weights for active synapses
        active_synapses = [s for s in self.synapses if s['to'] in neurons]

        for synapse in active_synapses:
            # Color based on weight sign
            if synapse['weight'] > 0:
                synapse['arrow'].set_color('blue')
            else:
                synapse['arrow'].set_color('red')

            # Thickness based on weight magnitude
            thickness = min(4, 1 + abs(synapse['weight']) * 2)
            synapse['arrow'].set_linewidth(thickness)
            synapse['arrow'].set_alpha(0.8)

            # Show weight label
            synapse['weight_label'].set_visible(True)

        plt.draw()
        plt.pause(0.01)

        print(f"      💡 Activating {len(neurons)} neurons...")
        print(f"      📊 Processing signals through {len(active_synapses)} synapses")

        # Update neuron activations
        activations = activation_data['data'][0]
        max_act = max(abs(activation_data['max']), abs(activation_data['min']), 0.001)

        for i, neuron in enumerate(neurons):
            if i < len(activations):
                norm_act = abs(activations[i]) / max_act
                intensity = min(1.0, norm_act)

                # Color based on layer position
                if layer_idx == len(self.layers) - 1:  # Output
                    color = (0.4 + 0.6*intensity, 1.0, 0.4 + 0.6*intensity)
                else:  # Hidden
                    color = (1.0, 0.4 + 0.6*intensity, 0.4 + 0.6*intensity)

                neuron['circle'].set_facecolor(color)
                neuron['activation'] = intensity

                print(f"         Neuron {i+1}: activation = {activations[i]:.4f}")

        plt.draw()
        plt.pause(0.01)
        time.sleep(self.animation_speed * 0.3)

        # Fade synapse highlights
        for synapse in active_synapses:
            synapse['arrow'].set_alpha(0.4)
            synapse['arrow'].set_color('gray')
            synapse['arrow'].set_linewidth(2)
            synapse['weight_label'].set_visible(False)

    def _animate_layer_backward_with_synapses(self, layer_name, layer_idx):
        """Animate backward propagation showing gradient flow through synapses."""
        if layer_name not in self.layer_gradients:
            return

        gradient_data = self.layer_gradients[layer_name]
        neurons = self.neurons[layer_name]

        print(f"\n   🎯 Layer {layer_idx + 1} ({layer_name})")
        print(f"      Computing: ∂Loss/∂weight = ∂Loss/∂activation × ∂activation/∂weight")

        # Update info panel
        info = f"""
BACKWARD PASS - Layer {layer_idx + 1}: {layer_name}
{'='*60}
GRADIENT FLOW THROUGH SYNAPSES:
  Gradients flow backward through synapses to update weights:

  synapse_gradient = neuron_gradient × previous_activation
  weight_update = learning_rate × synapse_gradient

GRADIENT STATISTICS:
  Mean:  {gradient_data['mean']:>8.6f}  (average gradient)
  Std:   {gradient_data['std']:>8.6f}  (gradient variation)
  Norm:  {gradient_data['norm']:>8.6f}  (gradient magnitude)
  Max:   {gradient_data['max']:>8.6f}  (largest gradient)
  Min:   {gradient_data['min']:>8.6f}  (smallest gradient)

⬅️  Watch red synapses carry gradients backward!
"""
        self.info_text.set_text(info)

        # Highlight active synapses for backward pass
        active_synapses = [s for s in self.synapses if s['to'] in neurons]

        for synapse in active_synapses:
            # Orange/red for backward pass
            synapse['arrow'].set_color('orangered')
            synapse['arrow'].set_linewidth(3)
            synapse['arrow'].set_alpha(0.8)
            synapse['weight_label'].set_visible(True)

        plt.draw()
        plt.pause(0.01)

        print(f"      🎯 Computing gradients for {len(neurons)} neurons")
        print(f"      📊 Flowing gradients through {len(active_synapses)} synapses")

        # Update neuron colors for gradients
        gradients = gradient_data['data'][0]
        max_grad = max(abs(gradient_data['max']), abs(gradient_data['min']), 0.001)

        for i, neuron in enumerate(neurons):
            if i < len(gradients):
                norm_grad = abs(gradients[i]) / max_grad
                intensity = min(1.0, norm_grad)

                # Red/orange for gradients
                color = (1.0, 0.2 + 0.5*intensity, 0.1*intensity)

                neuron['circle'].set_facecolor(color)
                neuron['gradient'] = intensity

                print(f"         Neuron {i+1}: gradient = {gradients[i]:.6f}")

        plt.draw()
        plt.pause(0.01)
        time.sleep(self.animation_speed * 0.3)

        # Fade synapse highlights
        for synapse in active_synapses:
            synapse['arrow'].set_alpha(0.4)
            synapse['arrow'].set_color('gray')
            synapse['arrow'].set_linewidth(2)
            synapse['weight_label'].set_visible(False)

    def _on_pause(self, event):
        """Handle pause button."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.label.set_text('▶ Resume')
            print("\n⏸️  Animation PAUSED - Click Resume or Step to continue")
        else:
            self.btn_pause.label.set_text('⏸ Pause')
            print("\n▶️  Animation RESUMED")

    def _on_step(self, event):
        """Handle step button."""
        if not self.is_paused:
            self.is_paused = True
            self.btn_pause.label.set_text('▶ Resume')
        print("\n⏭️  Stepping forward one layer...")

    def _on_reset(self, event):
        """Handle reset button."""
        self.mode = 'forward'
        self.current_step = 0
        # Reset all neuron colors
        for layer_neurons in self.neurons.values():
            for neuron in layer_neurons:
                if 'input' in neuron['id']:
                    neuron['circle'].set_facecolor('lightblue')
                elif 'output' in neuron['id']:
                    neuron['circle'].set_facecolor('lightgreen')
                else:
                    neuron['circle'].set_facecolor('lightcoral')
        plt.draw()
        print("\n🔄 Reset - Starting fresh from forward pass")

    def _on_speed_change(self, val):
        """Handle speed slider."""
        self.animation_speed = val
        print(f"\n⚡ Animation speed: {val:.1f}s per step")

    def stop_panel(self):
        """Stop the panel."""
        self.is_running = False
        for hook in self.hooks:
            hook.remove()
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        print("\n⏹️  Interactive Propagation Panel stopped")


def launch_propagation_panel(model, data_loader, device):
    """
    Launch the interactive propagation panel.

    Args:
        model: PyTorch model
        data_loader: DataLoader with samples
        device: Device to run on
    """
    print("\n" + "="*70)
    print("🚀 LAUNCHING INTERACTIVE PROPAGATION PANEL")
    print("="*70)
    print("\n📊 Preparing to visualize neuron-synapse interactions...")

    # Get a sample batch
    sample_batch = next(iter(data_loader))
    sample_input = sample_batch[0][:1].to(device)  # Single sample
    sample_target = sample_batch[1][:1].to(device)

    print(f"   ✓ Sample input shape: {sample_input.shape}")
    print(f"   ✓ Sample target: {sample_target.item()}")

    # Create and start panel
    panel = InteractivePropagationPanel(model, sample_input, sample_target)

    try:
        panel.start_panel()
    except KeyboardInterrupt:
        print("\n⏹️  Panel interrupted by user")
    finally:
        panel.stop_panel()


if __name__ == '__main__':
    print("Interactive Propagation Panel - Synapse Interaction Visualizer")
    print("Use launch_propagation_panel() to start with a real model")