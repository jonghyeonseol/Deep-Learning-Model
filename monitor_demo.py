#!/usr/bin/env python3
"""
Neural Network Monitoring and Visualization Demo

This script demonstrates how to monitor and visualize what your ANN model is doing:
- Perceptron structure visualization
- Layer-by-layer monitoring
- Activation function analysis
- Weight and gradient flow
- Real-time training monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import ConvNeuralNetwork, get_activation, get_available_activations
from utils import (
    PerceptronVisualizer, LayerMonitor, ActivationAnalyzer,
    RealTimeMonitor, CIFAR10DataLoader
)
import os


def demo_single_perceptron():
    """Demonstrate a single perceptron structure and operation."""
    print("=== Single Perceptron Demonstration ===")

    visualizer = PerceptronVisualizer()

    # Create a simple perceptron example
    weights = [0.7, -0.3, 0.5, 0.2]  # 4 input weights
    bias = -0.1
    inputs = [1.2, 0.8, -0.5, 0.3]  # Sample input values

    # Calculate output manually for different activations
    z = sum(w * x for w, x in zip(weights, inputs)) + bias

    activations_to_test = ['relu', 'sigmoid', 'tanh', 'swish']

    for activation_name in activations_to_test:
        try:
            activation_fn = get_activation(activation_name)
            output_tensor = activation_fn(torch.tensor(z))
            output = float(output_tensor.item())

            print(f"\n{activation_name.upper()} Activation:")
            print(f"  Weighted sum (z): {z:.3f}")
            print(f"  Output: {output:.3f}")

            # Visualize the perceptron
            save_path = f"./visualizations/perceptron_{activation_name}.png"
            visualizer.plot_single_perceptron(
                weights=weights,
                bias=bias,
                inputs=inputs,
                output=output,
                activation_name=activation_name,
                save_path=save_path
            )

        except Exception as e:
            print(f"Error with {activation_name}: {e}")


def demo_network_architecture():
    """Demonstrate neural network architecture visualization."""
    print("\n=== Network Architecture Visualization ===")

    visualizer = PerceptronVisualizer()

    # Create different network architectures
    networks = {
        'Simple CNN': ConvNeuralNetwork(input_channels=3, num_classes=10, activation='relu'),
        'GELU CNN': ConvNeuralNetwork(input_channels=3, num_classes=10, activation='gelu'),
        'Swish CNN': ConvNeuralNetwork(input_channels=3, num_classes=10, activation='swish')
    }

    for name, model in networks.items():
        print(f"\nVisualizing {name}...")
        save_path = f"./visualizations/architecture_{name.lower().replace(' ', '_')}.png"
        visualizer.plot_network_architecture(model, save_path=save_path)


def demo_layer_monitoring():
    """Demonstrate layer-by-layer monitoring during forward/backward pass."""
    print("\n=== Layer-by-Layer Monitoring ===")

    # Create model and data
    model = ConvNeuralNetwork(activation='swish')
    model.eval()

    # Create sample data
    batch_size = 8
    sample_input = torch.randn(batch_size, 3, 32, 32)
    sample_target = torch.randint(0, 10, (batch_size,))

    # Setup monitoring
    monitor = LayerMonitor()
    monitor.register_hooks(model)

    # Forward pass
    print("Performing forward pass...")
    output = model(sample_input)

    # Backward pass
    print("Performing backward pass...")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, sample_target)
    loss.backward()

    # Visualize layer activations
    print("Plotting layer activations...")
    monitor.plot_layer_activations(save_path="./visualizations/layer_activations.png")

    # Visualize gradient flow
    print("Plotting gradient flow...")
    monitor.plot_gradient_flow(save_path="./visualizations/gradient_flow.png")

    # Clean up
    monitor.clear_hooks()


def demo_activation_analysis():
    """Demonstrate activation function analysis and comparison."""
    print("\n=== Activation Function Analysis ===")

    analyzer = ActivationAnalyzer()

    # Analyze modern activation functions
    modern_activations = ['relu', 'gelu', 'swish', 'mish', 'silu', 'tanh', 'sigmoid']

    print("Plotting activation functions...")
    analyzer.plot_activation_functions(
        modern_activations,
        x_range=(-5, 5),
        save_path="./visualizations/activation_comparison.png"
    )

    # Analyze mathematical properties
    print("\nAnalyzing activation function properties...")
    properties = analyzer.analyze_activation_properties(modern_activations)

    print("\nActivation Function Properties:")
    print("-" * 80)
    for act_name, props in properties.items():
        if 'error' in props:
            print(f"{act_name.upper()}: Error - {props['error']}")
            continue

        print(f"\n{act_name.upper()}:")
        print(f"  Range: {props['range'][0]:.3f} to {props['range'][1]:.3f}")
        print(f"  Mean output: {props['mean_output']:.3f}")
        print(f"  Zero-centered: {props['zero_centered']}")
        print(f"  Monotonic: {props['monotonic']}")
        print(f"  Bounded: {props['bounded']}")
        print(f"  Saturating: {props['saturating']}")


def demo_weight_analysis():
    """Demonstrate weight distribution analysis."""
    print("\n=== Weight Distribution Analysis ===")

    model = ConvNeuralNetwork(activation='swish')

    plt.figure(figsize=(15, 10))

    layer_idx = 1
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weights = module.weight.detach().numpy().flatten()

            plt.subplot(2, 3, layer_idx)
            plt.hist(weights, bins=50, alpha=0.7, density=True)
            plt.title(f'{name}\nMean: {weights.mean():.4f}, Std: {weights.std():.4f}')
            plt.xlabel('Weight Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)

            layer_idx += 1
            if layer_idx > 6:  # Limit to 6 layers for display
                break

    plt.suptitle('Weight Distributions Across Layers', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig("./visualizations/weight_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()


def demo_real_time_monitoring():
    """Demonstrate real-time training monitoring (simulated)."""
    print("\n=== Real-time Training Monitoring (Simulated) ===")

    monitor = RealTimeMonitor()

    # Simulate training data
    print("Simulating training progress...")
    epochs = 3
    batches_per_epoch = 50

    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            # Simulate realistic training curves
            progress = (epoch * batches_per_epoch + batch) / (epochs * batches_per_epoch)

            # Simulated loss (decreasing with noise)
            base_loss = 2.0 * np.exp(-progress * 2) + 0.5
            noise = np.random.normal(0, 0.1)
            loss = max(0.1, base_loss + noise)

            # Simulated accuracy (increasing with noise)
            base_acc = 100 * (1 - np.exp(-progress * 3))
            acc_noise = np.random.normal(0, 2)
            accuracy = min(95, max(10, base_acc + acc_noise))

            # Simulated learning rate decay
            lr = 0.001 * (0.9 ** epoch)

            monitor.update(epoch + 1, batch + 1, loss, accuracy, lr)

    # Plot monitoring results
    monitor.plot_training_progress(save_path="./visualizations/training_monitor.png")


def demo_feature_maps():
    """Demonstrate feature map visualization for CNN layers."""
    print("\n=== Feature Map Visualization ===")

    model = ConvNeuralNetwork(activation='relu')
    model.eval()

    # Get a sample image
    try:
        data_loader = CIFAR10DataLoader(batch_size=1)
        train_loader, _, _ = data_loader.get_data_loaders()
        sample_input, sample_label = next(iter(train_loader))

        # Hook to capture feature maps
        feature_maps = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and len(output.shape) == 4:  # Conv output
                    feature_maps[name] = output.detach().cpu().numpy()
            return hook

        # Register hooks for conv layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn(name))

        # Forward pass
        with torch.no_grad():
            _ = model(sample_input)

        # Visualize feature maps
        if feature_maps:
            for layer_name, fmaps in feature_maps.items():
                # Show first 16 feature maps
                n_maps = min(16, fmaps.shape[1])

                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                fig.suptitle(f'Feature Maps - {layer_name}', fontsize=14, weight='bold')

                for i in range(n_maps):
                    row, col = i // 4, i % 4
                    axes[row, col].imshow(fmaps[0, i], cmap='viridis')
                    axes[row, col].set_title(f'Filter {i+1}')
                    axes[row, col].axis('off')

                # Hide unused subplots
                for i in range(n_maps, 16):
                    row, col = i // 4, i % 4
                    axes[row, col].axis('off')

                plt.tight_layout()
                plt.savefig(f"./visualizations/feature_maps_{layer_name.replace('.', '_')}.png",
                           dpi=300, bbox_inches='tight')
                plt.show()

        print("Feature map visualization completed!")

    except Exception as e:
        print(f"Feature map demo skipped (requires CIFAR-10 dataset): {e}")


def main():
    """Run all monitoring demonstrations."""
    print("Neural Network Monitoring and Visualization Demo")
    print("=" * 60)

    # Create visualizations directory
    os.makedirs("./visualizations", exist_ok=True)

    # Run demonstrations
    try:
        demo_single_perceptron()
    except Exception as e:
        print(f"Perceptron demo failed: {e}")

    try:
        demo_network_architecture()
    except Exception as e:
        print(f"Architecture demo failed: {e}")

    try:
        demo_layer_monitoring()
    except Exception as e:
        print(f"Layer monitoring demo failed: {e}")

    try:
        demo_activation_analysis()
    except Exception as e:
        print(f"Activation analysis demo failed: {e}")

    try:
        demo_weight_analysis()
    except Exception as e:
        print(f"Weight analysis demo failed: {e}")

    try:
        demo_real_time_monitoring()
    except Exception as e:
        print(f"Real-time monitoring demo failed: {e}")

    try:
        demo_feature_maps()
    except Exception as e:
        print(f"Feature maps demo failed: {e}")

    print("\n" + "=" * 60)
    print("Demo completed! Check the './visualizations' folder for saved plots.")
    print("\nGenerated visualizations:")
    print("- perceptron_*.png: Individual perceptron structures")
    print("- architecture_*.png: Network architectures")
    print("- layer_activations.png: Layer activation distributions")
    print("- gradient_flow.png: Gradient flow through layers")
    print("- activation_comparison.png: Activation function comparison")
    print("- weight_distributions.png: Weight distributions")
    print("- training_monitor.png: Training progress monitoring")
    print("- feature_maps_*.png: Convolutional feature maps")


if __name__ == '__main__':
    main()