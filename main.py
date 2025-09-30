#!/usr/bin/env python3
"""
Main script to train neural networks with different activation functions on CIFAR-10.
Usage: python main.py --activation all
"""

import argparse
import torch
import os
import time
from models import ConvNeuralNetwork, get_activation, get_available_activations
from utils import CIFAR10DataLoader, Trainer, Visualizer


def visualize_network(activation='relu'):
    """
    Visualize neural network structure in real-time.

    Args:
        activation (str): Activation function to use
    """
    print("🧠" + "="*60 + "🧠")
    print("    LIVE NEURAL NETWORK VISUALIZATION")
    print("🧠" + "="*60 + "🧠")
    print()
    print("📊 What you'll see:")
    print("   🔵 Blue circles = Input neurons")
    print("   🔴 Red circles = Hidden neurons")
    print("   🟢 Green circles = Output neurons")
    print("   ➡️  Arrows = Connections (weights)")
    print("   💡 Brightness = Neuron activation level")
    print("   📏 Arrow thickness = Weight strength")
    print()

    try:
        from utils.live_network_viz import LivePerceptronNetwork

        # Demo different network architectures
        demos = [
            {
                'name': 'Simple Network',
                'structure': [3, 4, 2],
                'description': '3 inputs → 4 hidden → 2 outputs'
            },
            {
                'name': 'Deep Network',
                'structure': [4, 6, 4, 3],
                'description': '4 inputs → 6 hidden → 4 hidden → 3 outputs'
            },
        ]

        input("👀 Press ENTER to start the live visualization... ")
        print()

        for i, demo in enumerate(demos, 1):
            print(f"🎯 Demo {i}/{len(demos)}: {demo['name']}")
            print(f"   Structure: {demo['description']}")
            print()

            network = LivePerceptronNetwork(demo['structure'])
            network.start_visualization()

            print("🎬 Starting animation (30 seconds)")
            print("   Watch the neurons and connections!")
            print()

            try:
                start_time = time.time()
                iteration = 0

                while time.time() - start_time < 30:
                    iteration += 1

                    # Create different data patterns
                    if iteration % 20 == 0:
                        input_data = [1.0] * demo['structure'][0]
                    elif iteration % 20 == 10:
                        input_data = [1.0 if i % 2 == 0 else 0.0 for i in range(demo['structure'][0])]
                    else:
                        import numpy as np
                        input_data = np.random.randn(demo['structure'][0])

                    network.animate_data_flow(input_data)
                    time.sleep(0.3)

            except KeyboardInterrupt:
                print("\n⏹️ Visualization stopped by user")

            network.stop_visualization()

            if i < len(demos):
                input("Press ENTER to continue to next demo... ")
                print()

        print("✅ Network visualization completed!")

    except ImportError as e:
        print(f"❌ Live visualization not available: {e}")
        print("   Make sure all dependencies are installed.")


def train_model(activation, epochs=10, batch_size=32, lr=0.001, enable_monitoring=False):
    """
    Train a model with specified activation function.

    Args:
        activation (str): Activation function name
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        enable_monitoring (bool): Enable live training monitoring

    Returns:
        dict: Training results
    """
    print(f"\n{'='*60}")
    print(f"Training model with {activation.upper()} activation")
    if enable_monitoring:
        print("📊 Live monitoring: ENABLED")
    print(f"{'='*60}")

    # Create data loader
    data_loader = CIFAR10DataLoader(batch_size=batch_size, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Print dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"Dataset: CIFAR-10")
    print(f"Train samples: {dataset_info['train_size']:,}")
    print(f"Validation samples: {dataset_info['val_size']:,}")
    print(f"Test samples: {dataset_info['test_size']:,}")
    print(f"Number of classes: {dataset_info['num_classes']}")

    # Create model
    model = ConvNeuralNetwork(
        input_channels=3,
        num_classes=10,
        activation=activation,
        dropout_rate=0.2
    )

    # Print model summary
    model.summary()

    # Create trainer (with live monitoring if requested)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'./checkpoints/{activation}'
    os.makedirs(save_dir, exist_ok=True)

    if enable_monitoring:
        try:
            from utils.live_trainer import LiveTrainer
            trainer = LiveTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                save_dir=save_dir,
                enable_live_monitoring=True
            )
            print("📺 Live monitoring initialized!")
        except ImportError:
            print("⚠️  Live monitoring not available, using standard trainer")
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                save_dir=save_dir
            )
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_dir=save_dir
        )

    # Configure training
    trainer.configure_optimizer('adam', lr=lr, weight_decay=1e-4)
    trainer.configure_scheduler('step', step_size=5, gamma=0.1)
    trainer.configure_criterion('crossentropy')

    # Train model
    trainer.train(epochs=epochs, early_stopping_patience=5, save_best=True)

    # Test model
    test_loss, test_acc = trainer.test()

    # Create visualizations
    visualizer = Visualizer(class_names=data_loader.classes)

    # Plot training history
    visualizer.plot_training_history(
        trainer.history,
        save_path=os.path.join(save_dir, 'training_history.png')
    )

    # Plot sample predictions
    visualizer.plot_model_predictions(
        model, test_loader, device, num_samples=8,
        denormalize_fn=data_loader.denormalize,
        save_path=os.path.join(save_dir, 'predictions.png')
    )

    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        model, test_loader, device,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )

    return {
        'activation': activation,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_val_acc,
        'model_parameters': model.get_num_parameters()
    }


def compare_activations(activations, epochs=10, batch_size=32, lr=0.001, enable_monitoring=False):
    """
    Compare different activation functions.

    Args:
        activations (list): List of activation function names
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        enable_monitoring (bool): Enable live monitoring
    """
    results = []

    for activation in activations:
        try:
            result = train_model(activation, epochs, batch_size, lr, enable_monitoring)
            results.append(result)
        except Exception as e:
            print(f"Error training with {activation}: {e}")
            continue

    # Display comparison results
    print(f"\n{'='*80}")
    print("ACTIVATION FUNCTION COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Activation':<12} {'Test Loss':<12} {'Test Acc (%)':<12} {'Val Acc (%)':<12} {'Parameters':<12}")
    print(f"{'-'*80}")

    for result in results:
        print(f"{result['activation']:<12} "
              f"{result['test_loss']:<12.4f} "
              f"{result['test_accuracy']:<12.2f} "
              f"{result['best_val_accuracy']:<12.2f} "
              f"{result['model_parameters']:<12,}")

    # Find best performing activation
    if results:
        best_result = max(results, key=lambda x: x['test_accuracy'])
        print(f"\nBest performing activation: {best_result['activation'].upper()}")
        print(f"Test accuracy: {best_result['test_accuracy']:.2f}%")


def main():
    # Get available activations dynamically
    available_activations = get_available_activations()
    activation_choices = available_activations + ['all', 'modern', 'classic']

    parser = argparse.ArgumentParser(description='Train neural networks with different activation functions')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=activation_choices,
                       help='Activation function to use (default: relu). Use "all" for all functions, "modern" for recent ones, "classic" for traditional ones.')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with fewer epochs for testing')
    parser.add_argument('--list-activations', action='store_true',
                       help='List all available activation functions')
    parser.add_argument('--visualize', action='store_true',
                       help='Show live network visualization (neurons and connections)')
    parser.add_argument('--monitor', action='store_true',
                       help='Enable live training monitoring with real-time plots')

    args = parser.parse_args()

    # Show network visualization if requested
    if args.visualize:
        visualize_network(args.activation)
        return

    # List activations if requested
    if args.list_activations:
        print("Available activation functions:")
        classic = ['relu', 'tanh', 'sigmoid', 'step', 'softmax']
        modern = ['gelu', 'swish', 'mish', 'silu', 'hardswish']
        others = ['leakyrelu', 'elu', 'prelu', 'selu']

        print(f"Classic: {', '.join(classic)}")
        print(f"Modern: {', '.join(modern)}")
        print(f"Others: {', '.join(others)}")
        print(f"All: {', '.join(available_activations)}")
        return

    # Adjust epochs for quick training
    if args.quick:
        args.epochs = 2
        print("Quick training mode: using 2 epochs")

    print("CIFAR-10 Neural Network Training")
    print(f"Arguments: {args}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Determine which activations to train
    if args.activation == 'all':
        activations = available_activations
        print(f"\nTraining with ALL activation functions: {activations}")
        compare_activations(activations, args.epochs, args.batch_size, args.lr, args.monitor)
    elif args.activation == 'modern':
        activations = ['gelu', 'swish', 'mish', 'silu', 'hardswish']
        print(f"\nTraining with MODERN activation functions: {activations}")
        compare_activations(activations, args.epochs, args.batch_size, args.lr, args.monitor)
    elif args.activation == 'classic':
        activations = ['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu']
        print(f"\nTraining with CLASSIC activation functions: {activations}")
        compare_activations(activations, args.epochs, args.batch_size, args.lr, args.monitor)
    else:
        # Train with single activation function
        print(f"\nTraining with {args.activation} activation")
        result = train_model(args.activation, args.epochs, args.batch_size, args.lr, args.monitor)
        print(f"\nFinal results:")
        print(f"Test accuracy: {result['test_accuracy']:.2f}%")
        print(f"Test loss: {result['test_loss']:.4f}")


if __name__ == '__main__':
    main()