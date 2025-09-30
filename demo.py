#!/usr/bin/env python3
"""
Demo script to showcase the neural network components without full training.
"""

import torch
import numpy as np
from models import NeuralNetwork, ConvNeuralNetwork, get_activation, get_available_activations
from utils import CIFAR10DataLoader


def demo_activation_functions():
    """Demonstrate different activation functions."""
    print("=== Activation Functions Demo ===")

    # Create test input
    x = torch.randn(5, 10)
    print(f"Input tensor shape: {x.shape}")
    print(f"Input sample: {x[0, :5].detach().numpy()}")

    # Demo a subset of activations for clarity
    demo_activations = ['relu', 'gelu', 'tanh', 'sigmoid', 'swish', 'mish']

    print(f"\nTesting {len(demo_activations)} activation functions:")
    print(f"Available: {', '.join(get_available_activations())}")

    for act_name in demo_activations:
        try:
            activation = get_activation(act_name)
            output = activation(x)
            print(f"\n{act_name.upper()} output sample: {output[0, :5].detach().numpy()}")
        except Exception as e:
            print(f"\n{act_name.upper()} failed: {e}")


def demo_neural_networks():
    """Demonstrate neural network architectures."""
    print("\n=== Neural Network Architectures Demo ===")

    # Fully connected network
    print("\n1. Fully Connected Network:")
    fc_net = NeuralNetwork(
        input_size=784,  # 28x28 flattened
        hidden_sizes=[512, 256, 128],
        output_size=10,
        activation='relu'
    )
    fc_net.summary()

    # Test forward pass
    test_input = torch.randn(32, 784)
    output = fc_net(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Convolutional network
    print("\n2. Convolutional Network (for CIFAR-10):")
    conv_net = ConvNeuralNetwork(
        input_channels=3,
        num_classes=10,
        activation='gelu'
    )
    conv_net.summary()

    # Test forward pass
    test_input = torch.randn(32, 3, 32, 32)  # CIFAR-10 format
    output = conv_net(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")


def demo_data_loader():
    """Demonstrate CIFAR-10 data loader."""
    print("\n=== CIFAR-10 Data Loader Demo ===")

    try:
        # Create data loader
        data_loader = CIFAR10DataLoader(batch_size=8, validation_split=0.1)

        # Get dataset info
        info = data_loader.get_dataset_info()
        print("Dataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Get sample batch
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()
        images, labels = next(iter(train_loader))

        print(f"\nSample batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Class names: {[data_loader.classes[label] for label in labels]}")

        # Show class distribution
        distribution = data_loader.get_class_distribution('train')
        print(f"\nTraining set class distribution (first 5 classes):")
        for i, (class_name, count) in enumerate(list(distribution.items())[:5]):
            print(f"  {class_name}: {count}")

    except Exception as e:
        print(f"Note: CIFAR-10 dataset not available (requires download): {e}")
        print("This is normal - the dataset will be downloaded on first use.")


def demo_model_comparison():
    """Compare models with different activation functions."""
    print("\n=== Model Comparison Demo ===")

    # Test a subset of activations for speed
    test_activations = ['relu', 'gelu', 'swish', 'mish', 'sigmoid']

    for activation in test_activations:
        try:
            model = ConvNeuralNetwork(activation=activation)
            num_params = model.get_num_parameters()

            # Test inference speed (rough estimate)
            test_input = torch.randn(32, 3, 32, 32)

            # Warm up
            with torch.no_grad():
                _ = model(test_input)

            # Time inference
            import time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(test_input)
            inference_time = (time.time() - start_time) / 100 * 1000  # ms per inference

            print(f"{activation.upper()}: {num_params:,} parameters, "
                  f"~{inference_time:.2f}ms per batch inference")
        except Exception as e:
            print(f"{activation.upper()}: Error - {e}")


def main():
    print("Neural Network Implementation Demo")
    print("=" * 50)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Run demos
    demo_activation_functions()
    demo_neural_networks()
    demo_data_loader()
    demo_model_comparison()

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTo run full training, use:")
    print("  python main.py --activation relu --epochs 10")
    print("  python main.py --activation all --epochs 5")


if __name__ == '__main__':
    main()